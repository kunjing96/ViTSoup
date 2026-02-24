# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import random
import datetime
import time
import weakref

from collections import OrderedDict, abc
from typing import Any, Dict, List, Set, Union
from contextlib import ExitStack

import torch
from torch import nn
from timm.utils.model import unwrap_model

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
    DatasetEvaluator,
    print_csv_format,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger, log_every_n_seconds, _log_api_usage
from detectron2.utils.comm import get_world_size
from detectron2.evaluation import inference_context
from detectron2.config import CfgNode as CN

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_maskformer2_super_config
)


def sample_configs(search_space):
    config = CN()
    # BACKBONE
    if search_space.BACKBONE.IS_USED:
        config.BACKBONE = CN()
        config.BACKBONE.EMBED_DIM = random.choice(search_space.BACKBONE.EMBED_DIM)
        config.BACKBONE.DEPTHS = [ random.choice(depth) for depth in search_space.BACKBONE.DEPTHS ]
        config.BACKBONE.NUM_HEADS = []
        config.BACKBONE.WINDOW_SIZE = []
        config.BACKBONE.MLP_RATIO = []
        for i, depth in enumerate(config.BACKBONE.DEPTHS):
            for _ in range(depth):
                config.BACKBONE.NUM_HEADS.append(random.choice(search_space.BACKBONE.NUM_HEADS[i]))
                config.BACKBONE.WINDOW_SIZE.append(random.choice(search_space.BACKBONE.WINDOW_SIZE))
                config.BACKBONE.MLP_RATIO.append(random.choice(search_space.BACKBONE.MLP_RATIO))
    else:
        config.BACKBONE = None
    # SEM_SEG_HEAD
    if search_space.SEM_SEG_HEAD.IS_USED:
        config.SEM_SEG_HEAD = CN()
        config.SEM_SEG_HEAD.IN_FEATURES = search_space.SEM_SEG_HEAD.IN_FEATURES
        while True:
            enc_in_features = random.choice(search_space.SEM_SEG_HEAD.ENC_IN_FEATURES)
            for feature in enc_in_features:
                if feature not in search_space.SEM_SEG_HEAD.IN_FEATURES:
                    break
            else:
                break
        config.SEM_SEG_HEAD.ENC_IN_FEATURES = enc_in_features
        config.SEM_SEG_HEAD.CONVS_DIM = random.choice(search_space.SEM_SEG_HEAD.CONVS_DIM)
        config.SEM_SEG_HEAD.ENC_DEPTHS = random.choice(search_space.SEM_SEG_HEAD.ENC_DEPTHS)
        config.SEM_SEG_HEAD.ENC_N_POINTS = []
        config.SEM_SEG_HEAD.ENC_N_HEADS = []
        config.SEM_SEG_HEAD.ENC_MLP_RATIO = []
        for _ in range(config.SEM_SEG_HEAD.ENC_DEPTHS):
            config.SEM_SEG_HEAD.ENC_N_POINTS.append(random.choice(search_space.SEM_SEG_HEAD.ENC_N_POINTS))
            config.SEM_SEG_HEAD.ENC_N_HEADS.append(random.choice(search_space.SEM_SEG_HEAD.ENC_N_HEADS))
            config.SEM_SEG_HEAD.ENC_MLP_RATIO.append(random.choice(search_space.SEM_SEG_HEAD.ENC_MLP_RATIO))
        config.SEM_SEG_HEAD.MASK_DIM = random.choice(search_space.SEM_SEG_HEAD.MASK_DIM)
    else:
        config.SEM_SEG_HEAD = None
    # MASK_FORMER
    if search_space.MASK_FORMER.IS_USED:
        config.MASK_FORMER = CN()
        config.MASK_FORMER.DEC_IN_FEATURES = random.choice(search_space.MASK_FORMER.DEC_IN_FEATURES)
        config.MASK_FORMER.HIDDEN_DIM = random.choice(search_space.MASK_FORMER.HIDDEN_DIM)
        config.MASK_FORMER.DEC_DEPTHS = random.choice(search_space.MASK_FORMER.DEC_DEPTHS)
        config.MASK_FORMER.DEC_CROSS_N_HEADS = []
        config.MASK_FORMER.DEC_SELF_N_HEADS = []
        config.MASK_FORMER.DEC_FFN_MLP_RATIO = []
        config.MASK_FORMER.MULTI_SCALE_PER_LAYER = []
        for _ in range(config.MASK_FORMER.DEC_DEPTHS-1):
            config.MASK_FORMER.DEC_CROSS_N_HEADS.append(random.choice(search_space.MASK_FORMER.DEC_N_HEADS))
            config.MASK_FORMER.DEC_SELF_N_HEADS.append(random.choice(search_space.MASK_FORMER.DEC_N_HEADS))
            config.MASK_FORMER.DEC_FFN_MLP_RATIO.append(random.choice(search_space.MASK_FORMER.DEC_MLP_RATIO))
        config.MASK_FORMER.MULTI_SCALE_PER_LAYER = (config.MASK_FORMER.DEC_IN_FEATURES * ((config.MASK_FORMER.DEC_DEPTHS-1-1) // len(config.MASK_FORMER.DEC_IN_FEATURES) + 1))[:config.MASK_FORMER.DEC_DEPTHS-1][::-1]
    else:
        config.MASK_FORMER = None
    return config


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, data in enumerate(data_loader):
        batch_size = len(data)
        for i in range(50):
            model(data)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(30):
            model(data)
        torch.cuda.synchronize()
        tic2 = time.time()
        _throughput = 30 * batch_size / (tic2 - tic1)
        return _throughput, batch_size


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], mode, sub_config, search_space
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger('mask2former')
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        if mode != "":
            if mode == 'super':
                config = sample_configs(search_space)
            else:
                config = sub_config
            model_module = unwrap_model(model)
            if isinstance(model_module, SemanticSegmentorWithTTA):
                model_module.model.set_sample_config(config=config)
            else:
                model_module.set_sample_config(config=config)

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self._hooks = []
        self.iter = 0
        self.start_iter = 0
        self.max_iter = None
        self.storage = None
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.model = model

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def run_step(self):
        self._trainer.iter = self.iter
        if self.cfg.MODE != "":
            if self.cfg.MODE == 'super':
                config = sample_configs(self.cfg.SEARCH_SPACE)
            else:
                config = self.cfg.SUBNET
            model_module = unwrap_model(self.model)
            model_module.set_sample_config(config=config)
        self._trainer.run_step()

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger('mask2former')
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator, cfg.MODE, cfg.SUBNET, cfg.SEARCH_SPACE)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_super_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    logger = logging.getLogger('mask2former')
    data_loader_val_dict = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader_val_dict[dataset_name] = Trainer.build_test_loader(cfg, dataset_name)
    model = trainer.model
    if ('Super' in cfg.MODEL.BACKBONE.NAME+cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME+cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME):
        if cfg.MODE == 'super':
            model_module = unwrap_model(model)
            model_module.set_sample_config(cfg.MIN_SUBNET)
            min_n_parameters = model_module.params()
            min_flops = model_module.flops(1024, 1024)
            min_throughput, _ = throughput(data_loader_val_dict[cfg.DATASETS.TEST[0]], model_module, None)
            model_module = unwrap_model(model)
            model_module.set_sample_config(cfg.MAX_SUBNET)
            max_n_parameters = model_module.params()
            max_flops = model_module.flops(1024, 1024)
            max_throughput, _ = throughput(data_loader_val_dict[cfg.DATASETS.TEST[0]], model_module, None)
            logger.info(f"number of params: {min_n_parameters / 1e6}M ~ {max_n_parameters / 1e6}M")
            logger.info(f"number of FLOPs: {min_flops / 1e9}G ~ {max_flops / 1e9}G")
            logger.info(f"number of throughput: {min_throughput}imgs/s ~ {max_throughput}imgs/s")
        else:
            model_module = unwrap_model(model)
            model_module.set_sample_config(cfg.SUBNET)
            n_parameters = model_module.params()
            flops = model_module.flops(1024, 1024)
            _throughput, _ = throughput(data_loader_val_dict[cfg.DATASETS.TEST[0]], model_module, None)
            logger.info(f"number of params: {n_parameters / 1e6}M")
            logger.info(f"number of FLOPs: {flops / 1e9}G")
            logger.info(f"number of throughput: {_throughput}imgs/s")
    else:
        model_module = unwrap_model(model)
        n_parameters = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_module, 'flops'):
            flops = model_module.flops(1024, 1024)
            logger.info(f"number of FLOPs: {flops / 1e9}G")
        _throughput, _ = throughput(data_loader_val_dict[cfg.DATASETS.TEST[0]], model_module, None)
        logger.info(f"number of throughput: {_throughput}imgs/s")

    model_module.train()
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
