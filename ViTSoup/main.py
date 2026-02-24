# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEmaV2
from timm.utils.model import unwrap_model
from timm.models import create_model

from yacs.config import CfgNode as CN

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--pretrained', help='load pretrained model from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def sample_configs(search_space):
    config = CN()
    config.EMBED_DIM = random.choice(search_space.EMBED_DIM)
    config.DEPTHS = [ random.choice(depth) for depth in search_space.DEPTHS ]
    config.NUM_HEADS = []
    config.WINDOW_SIZE = []
    config.MLP_RATIO = []
    for i, depth in enumerate(config.DEPTHS):
        for _ in range(depth):
            config.NUM_HEADS.append(random.choice(search_space.NUM_HEADS[i]))
            config.WINDOW_SIZE.append(random.choice(search_space.WINDOW_SIZE))
            config.MLP_RATIO.append(random.choice(search_space.MLP_RATIO))
    return config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=('super' in config.MODEL.TYPE))

    if 'super' in config.MODEL.TYPE:
        if config.MODE == 'super':
            model_module = unwrap_model(model)
            model_module.set_sample_config(config.MIN_SUBNET)
            min_n_parameters = model_module.params()
            min_flops = model_module.flops()
            min_throughput, _ = throughput(data_loader_val, model_module, logger)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config.MAX_SUBNET)
            max_n_parameters = model_module.params()
            max_flops = model_module.flops()
            max_throughput, _ = throughput(data_loader_val, model_module, logger)
            logger.info(f"number of params: {min_n_parameters / 1e6}M ~ {max_n_parameters / 1e6}M")
            logger.info(f"number of FLOPs: {min_flops / 1e9}G ~ {max_flops / 1e9}G")
            logger.info(f"number of throughput: {min_throughput}imgs/s ~ {max_throughput}imgs/s")
        else:
            model_module = unwrap_model(model)
            model_module.set_sample_config(config.SUBNET)
            n_parameters = model_module.params()
            flops = model_module.flops()
            _throughput, _ = throughput(data_loader_val, model_module, logger)
            logger.info(f"number of params: {n_parameters / 1e6}M")
            logger.info(f"number of FLOPs: {flops / 1e9}G")
            logger.info(f"number of throughput: {_throughput}imgs/s")
    else:
        model_module = unwrap_model(model)
        n_parameters = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_module, 'flops'):
            flops = model_module.flops()
            logger.info(f"number of FLOPs: {flops / 1e9}G")
        _throughput, _ = throughput(data_loader_val, model_module, logger)
        logger.info(f"number of throughput: {_throughput}imgs/s")

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        if 'super' in config.MODEL.TYPE:
            if config.MODE == 'super':
                subnet = sample_configs(config.SUPER)
            else:
                subnet = config.SUBNET
            model_module = unwrap_model(model)
            model_module.set_sample_config(subnet)
        _throughput, batch_size = throughput(data_loader_val, model_module, logger)
        logger.info(f"batch_size {batch_size} throughput {_throughput}")
        return

    # teacher model
    if config.TRAIN.TEACHER_MODEL is not None and config.TRAIN.TEACHER_MODEL != "":
        if config.TRAIN.TEACHER_MODEL != 'inplace':
            teacher_model = create_model(
                config.TRAIN.TEACHER_MODEL,
                pretrained=True,
                num_classes=config.MODEL.NUM_CLASSES,
            )
            teacher_model.cuda()
        else:
            if config.TRAIN.MODEL_EMA:
                teacher_model = ModelEmaV2(model_without_ddp, ecay=config.TRAIN.MODEL_EMA_DECAY)
                teacher_model.set(model_without_ddp)
            else:
                teacher_model = None
        if config.TRAIN.TEACHER_TYPE == 'hard':
            if config.MODEL.LABEL_SMOOTHING > 0.:
                teacher_loss = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
            else:
                teacher_loss = torch.nn.CrossEntropyLoss()
        elif config.TRAIN.TEACHER_TYPE == 'soft':
            teacher_loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)
        else:
            raise ValueError('Undefined distillation type {}'.format(config.TRAIN.TEACHER_TYPE))
    else:
        teacher_model = None
        teacher_loss = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, teacher_model=teacher_model, teacher_loss=teacher_loss)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, teacher_model=None, teacher_loss=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    if teacher_model is None:
        teacher_model = model

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if 'super' in config.MODEL.TYPE:
            if config.MODE == 'super':
                subnet = sample_configs(config.SUPER)
            else:
                subnet = config.SUBNET
            model_module = unwrap_model(model)
            model_module.set_sample_config(subnet)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        if config.TRAIN.TEACHER_MODEL is not None and config.TRAIN.TEACHER_MODEL != "":
            if config.TRAIN.TEACHER_MODEL != 'inplace':
                teacher_model.eval()
                with torch.no_grad():
                    teacher_output = teacher_model(samples).detach()
            else:
                teacher_model.eval()
                if 'super' in config.MODEL.TYPE:
                    if config.MODE == 'super':
                        subnet = config.MAX_SUBNET
                    else:
                        subnet = config.SUBNET
                    model_module = unwrap_model(teacher_model)
                    model_module.set_sample_config(subnet)
                with torch.no_grad():
                    teacher_output = teacher_model(samples).detach()
                teacher_model.train()
            if config.TRAIN.TEACHER_TYPE == 'hard':
                _, teacher_label = teacher_output.topk(1, 1, True, True)
                loss = (1 - config.TRAIN.TEACHER_ALPHA) * criterion(outputs, targets) + config.TRAIN.TEACHER_ALPHA * teacher_loss(outputs, teacher_label.squeeze())
            elif config.TRAIN.TEACHER_TYPE == 'soft':
                T = config.TRAIN.TEACHER_TAU
                log_prob = torch.nn.functional.log_softmax(outputs / T, dim=1)
                teacher_log_prob = torch.nn.functional.log_softmax(teacher_output / T, dim=1)
                loss = (1 - config.TRAIN.TEACHER_ALPHA) * criterion(outputs, targets) + config.TRAIN.TEACHER_ALPHA * teacher_loss(log_prob, teacher_log_prob) * T * T / outputs.numel()
            else:
                raise ValueError('Undefined distillation type {}'.format(config.TRAIN.TEACHER_TYPE))
        else:
            loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if isinstance(teacher_model, ModelEmaV2):
            teacher_model.update(unwrap_model(model))

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, subnet=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    if 'super' in config.MODEL.TYPE:
        if subnet is None:
            if config.MODE == 'super':
                subnet = config.MAX_SUBNET # sample_configs(config.SUPER)
            else:
                subnet = config.SUBNET
        model_module = unwrap_model(model)
        model_module.set_sample_config(subnet)
        n_parameters = model_module.params()
        flops = model_module.flops()
        _throughput, _ = throughput(data_loader, model_module, logger)
        logger.info(f"sampled model config: {subnet}")
        logger.info(f"sampled model params: {n_parameters / 1e6}M")
        logger.info(f"sampled model FLOPs: {flops / 1e9}")
        logger.info(f"sampled model throughput: {_throughput}imgs/s")

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        _throughput = 30 * batch_size / (tic2 - tic1)
        return _throughput, batch_size


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
