import os
import time
import random
import logging
from collections import OrderedDict
import detectron2.utils.comm as comm

import torch
from timm.utils.model import unwrap_model

from detectron2.engine import launch, default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN, get_cfg
from detectron2.engine.defaults import create_ddp_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import print_csv_format
from detectron2.utils.comm import all_gather

from train_net import Trainer, sample_configs, inference_on_dataset, throughput
from mask2former import add_maskformer2_config, add_maskformer2_super_config, add_search_config


def cfg2cand(cfg):
    cand = []
    # BACKBONE
    if cfg.BACKBONE is not None:
        cand.append(cfg.BACKBONE.EMBED_DIM)
        cand.append(tuple(cfg.BACKBONE.DEPTHS))
        cand.append(tuple(cfg.BACKBONE.NUM_HEADS))
        cand.append(tuple(cfg.BACKBONE.WINDOW_SIZE))
        cand.append(tuple(cfg.BACKBONE.MLP_RATIO))
    # SEM_SEG_HEAD
    if cfg.SEM_SEG_HEAD is not None:
        cand.append(tuple(cfg.SEM_SEG_HEAD.IN_FEATURES))
        cand.append(tuple(cfg.SEM_SEG_HEAD.ENC_IN_FEATURES))
        cand.append(cfg.SEM_SEG_HEAD.CONVS_DIM)
        cand.append(cfg.SEM_SEG_HEAD.ENC_DEPTHS)
        cand.append(tuple(cfg.SEM_SEG_HEAD.ENC_N_HEADS))
        cand.append(tuple(cfg.SEM_SEG_HEAD.ENC_N_POINTS))
        cand.append(tuple(cfg.SEM_SEG_HEAD.ENC_MLP_RATIO))
        cand.append(cfg.SEM_SEG_HEAD.MASK_DIM)
    # MASK_FORMER
    if cfg.MASK_FORMER is not None:
        cand.append(tuple(cfg.MASK_FORMER.DEC_IN_FEATURES))
        cand.append(cfg.MASK_FORMER.HIDDEN_DIM)
        cand.append(cfg.MASK_FORMER.DEC_DEPTHS)
        cand.append(tuple(cfg.MASK_FORMER.DEC_CROSS_N_HEADS))
        cand.append(tuple(cfg.MASK_FORMER.DEC_SELF_N_HEADS))
        cand.append(tuple(cfg.MASK_FORMER.DEC_FFN_MLP_RATIO))
        cand.append(tuple(cfg.MASK_FORMER.MULTI_SCALE_PER_LAYER))
    return tuple(cand)

def cand2cfg(cand):
    offset = 0
    cfg = CN()
    if len(cand) in [5, 12, 13, 20]:
        # BACKBONE
        cfg.BACKBONE = CN()
        cfg.BACKBONE.EMBED_DIM = cand[0]
        cfg.BACKBONE.DEPTHS = list(cand[1])
        cfg.BACKBONE.NUM_HEADS = list(cand[2])
        cfg.BACKBONE.WINDOW_SIZE = list(cand[3])
        cfg.BACKBONE.MLP_RATIO = list(cand[4])
    else:
        cfg.BACKBONE = None
    if len(cand) in [8, 13, 15, 20]:
        if len(cand) in [13, 20]:
            offset += 5
        # SEM_SEG_HEAD
        cfg.SEM_SEG_HEAD = CN()
        cfg.SEM_SEG_HEAD.IN_FEATURES = list(cand[offset+0])
        cfg.SEM_SEG_HEAD.ENC_IN_FEATURES = list(cand[offset+1])
        cfg.SEM_SEG_HEAD.CONVS_DIM = cand[offset+2]
        cfg.SEM_SEG_HEAD.ENC_DEPTHS = cand[offset+3]
        cfg.SEM_SEG_HEAD.ENC_N_HEADS = list(cand[offset+4])
        cfg.SEM_SEG_HEAD.ENC_N_POINTS = list(cand[offset+5])
        cfg.SEM_SEG_HEAD.ENC_MLP_RATIO = list(cand[offset+6])
        cfg.SEM_SEG_HEAD.MASK_DIM = cand[offset+7]
    else:
        cfg.SEM_SEG_HEAD = None
    if len(cand) in [7, 12, 15, 20]:
        if len(cand) in [15, 20]:
            offset += 8
        # MASK_FORMER
        cfg.MASK_FORMER = CN()
        cfg.MASK_FORMER.DEC_IN_FEATURES = list(cand[offset+0])
        cfg.MASK_FORMER.HIDDEN_DIM = cand[offset+1]
        cfg.MASK_FORMER.DEC_DEPTHS = cand[offset+2]
        cfg.MASK_FORMER.DEC_CROSS_N_HEADS = list(cand[offset+3])
        cfg.MASK_FORMER.DEC_SELF_N_HEADS = list(cand[offset+4])
        cfg.MASK_FORMER.DEC_FFN_MLP_RATIO = list(cand[offset+5])
        cfg.MASK_FORMER.MULTI_SCALE_PER_LAYER = list(cand[offset+6])
    else:
        cfg.MASK_FORMER = None
    return cfg

class EvolutionSearcher(object):

    def __init__(self, cfg, model, val_loader_dict, logger=None):
        self.model = model
        self.cfg = cfg
        self.max_epochs = cfg.SEARCH.MAX_EPOCHS
        self.select_num = cfg.SEARCH.SELECT_NUM
        self.population_num = cfg.SEARCH.POPULATION_NUM
        self.m_prob = cfg.SEARCH.M_PROB
        self.d_prob = cfg.SEARCH.D_PROB
        self.crossover_num = cfg.SEARCH.CROSSOVER_NUM
        self.mutation_num = cfg.SEARCH.MUTATION_NUM
        self.parameters_limits = cfg.SEARCH.PARAM_LIMITS
        self.min_parameters_limits = cfg.SEARCH.MIN_PARAM_LIMITS
        self.flops_limits = cfg.SEARCH.FLOPS_LIMITS
        self.min_flops_limits = cfg.SEARCH.MIN_FLOPS_LIMITS
        self.throughput_limits = cfg.SEARCH.THROUGHPUT_LIMITS
        self.min_throughput_limits = cfg.SEARCH.MIN_THROUGHPUT_LIMITS
        self.val_loader_dict = val_loader_dict
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = cfg.SEARCH.RESUME
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = cfg.SEARCH_SPACE
        self.output_dir = cfg.OUTPUT_DIR
        self.logger = logger

    def save_checkpoint(self):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        self.logger.info('save checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.top_accuracies = info['top_accuracies']
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.logger.info('load checkpoint from {}'.format(self.checkpoint_path))
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        subnet = cand2cfg(cand)
        model_module = unwrap_model(self.model)
        model_module.set_sample_config(subnet)
        n_parameters = model_module.params()
        flops = model_module.flops(1024, 1024)
        _throughput, _ = throughput(self.val_loader_dict[self.cfg.DATASETS.TEST[0]], model_module, None)
        info['params'] =  n_parameters / 10.**6
        info['flops'] =  flops / 10.**9
        info['throughput'] = _throughput

        if self.parameters_limits is not None and info['params'] > self.parameters_limits:
            self.logger.info('parameters limit exceed')
            return False

        if self.min_parameters_limits is not None and info['params'] < self.min_parameters_limits:
            self.logger.info('under minimum parameters limit')
            return False

        if self.flops_limits is not None and info['flops'] > self.flops_limits:
            self.logger.info('flops limit exceed')
            return False

        if self.min_flops_limits is not None and info['flops'] < self.min_flops_limits:
            self.logger.info('under minimum flops limit')
            return False

        if self.throughput_limits is not None and info['throughput'] > self.throughput_limits:
            self.logger.info('throughput limit exceed')
            return False

        if self.min_throughput_limits is not None and info['throughput'] < self.min_throughput_limits:
            self.logger.info('under minimum throughput limit')
            return False

        self.logger.info("rank: {} {} {} {} {}".format(comm.get_rank(), cand, info['params'], info['flops'], info['throughput']))
        res = validate(self.cfg, self.val_loader_dict, self.model, subnet=subnet)

        info['res'] = res
        info['acc'] = 0
        for k in res:
            if 'panoptic_seg' in res[k]:
                info['acc'] += res[k]['panoptic_seg']['PQ']
            if 'segm' in res[k]:
                info['acc'] += res[k]['segm']['AP']
            if 'sem_seg' in res[k]:
                info['acc'] += res[k]['sem_seg']['mIoU']

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        self.logger.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        return cfg2cand(sample_configs(self.choices))

    def get_random(self, num):
        self.logger.info('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            self.logger.info('random {}/{}'.format(len(self.candidates), num))
        self.logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, d_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            subnet = cand2cfg(cand)

            # BACKBONE
            if subnet.BACKBONE is not None:
                # EMBED_DIM
                if random.random() < m_prob:
                    subnet.BACKBONE.EMBED_DIM = random.choice(self.choices.BACKBONE.EMBED_DIM)
                # DEPTHS
                for i in range(len(subnet.BACKBONE.DEPTHS)):
                    if random.random() < d_prob:
                        old_depth = subnet.BACKBONE.DEPTHS[i]
                        left = sum(subnet.BACKBONE.DEPTHS[:i])
                        right = sum(subnet.BACKBONE.DEPTHS[:i+1])
                        subnet.BACKBONE.DEPTHS[i] = random.choice(self.choices.BACKBONE.DEPTHS[i])
                        if subnet.BACKBONE.DEPTHS[i] > old_depth:
                            subnet.BACKBONE.NUM_HEADS = subnet.BACKBONE.NUM_HEADS[:right] + [random.choice(self.choices.BACKBONE.NUM_HEADS[i]) for _ in range(subnet.BACKBONE.DEPTHS[i] - old_depth)] + subnet.BACKBONE.NUM_HEADS[right:]
                            subnet.BACKBONE.WINDOW_SIZE = subnet.BACKBONE.WINDOW_SIZE[:right] + [random.choice(self.choices.BACKBONE.WINDOW_SIZE) for _ in range(subnet.BACKBONE.DEPTHS[i] - old_depth)] + subnet.BACKBONE.WINDOW_SIZE[right:]
                            subnet.BACKBONE.MLP_RATIO = subnet.BACKBONE.MLP_RATIO[:right] + [random.choice(self.choices.BACKBONE.MLP_RATIO) for _ in range(subnet.BACKBONE.DEPTHS[i] - old_depth)] + subnet.BACKBONE.MLP_RATIO[right:]
                        else:
                            subnet.BACKBONE.NUM_HEADS = subnet.BACKBONE.NUM_HEADS[:left] + subnet.BACKBONE.NUM_HEADS[left:right][:subnet.BACKBONE.DEPTHS[i]] + subnet.BACKBONE.NUM_HEADS[right:]
                            subnet.BACKBONE.WINDOW_SIZE = subnet.BACKBONE.WINDOW_SIZE[:left] + subnet.BACKBONE.WINDOW_SIZE[left:right][:subnet.BACKBONE.DEPTHS[i]] + subnet.BACKBONE.WINDOW_SIZE[right:]
                            subnet.BACKBONE.MLP_RATIO = subnet.BACKBONE.MLP_RATIO[:left] + subnet.BACKBONE.MLP_RATIO[left:right][:subnet.BACKBONE.DEPTHS[i]] + subnet.BACKBONE.MLP_RATIO[right:]
                # NUM_HEADS, WINDOW_SIZE, MLP_RATIO
                for i in range(len(subnet.BACKBONE.DEPTHS)):
                    for j in range(subnet.BACKBONE.DEPTHS[i]):
                        l = sum(subnet.BACKBONE.DEPTHS[:i]) + j
                        if random.random() < m_prob:
                            subnet.BACKBONE.NUM_HEADS[l] = random.choice(self.choices.BACKBONE.NUM_HEADS[i])
                        if random.random() < m_prob:
                            subnet.BACKBONE.WINDOW_SIZE[l] = random.choice(self.choices.BACKBONE.WINDOW_SIZE)
                        if random.random() < m_prob:
                            subnet.BACKBONE.MLP_RATIO[l] = random.choice(self.choices.BACKBONE.MLP_RATIO)

            # SEM_SEG_HEAD
            if subnet.SEM_SEG_HEAD is not None:
                # IN_FEATURES
                subnet.SEM_SEG_HEAD.IN_FEATURES = self.choices.SEM_SEG_HEAD.IN_FEATURES
                # ENC_IN_FEATURES
                if random.random() < m_prob:
                    while True:
                        enc_in_features = random.choice(self.choices.SEM_SEG_HEAD.ENC_IN_FEATURES)
                        for feature in enc_in_features:
                            if feature not in self.choices.SEM_SEG_HEAD.IN_FEATURES:
                                break
                        else:
                            break
                    subnet.SEM_SEG_HEAD.ENC_IN_FEATURES = enc_in_features
                # CONVS_DIM
                if random.random() < m_prob:
                    subnet.SEM_SEG_HEAD.CONVS_DIM = random.choice(self.choices.SEM_SEG_HEAD.CONVS_DIM)
                # ENC_DEPTHS
                if random.random() < d_prob:
                    old_depth = subnet.SEM_SEG_HEAD.ENC_DEPTHS
                    subnet.SEM_SEG_HEAD.ENC_DEPTHS = random.choice(self.choices.SEM_SEG_HEAD.ENC_DEPTHS)
                    if subnet.SEM_SEG_HEAD.ENC_DEPTHS > old_depth:
                        subnet.SEM_SEG_HEAD.ENC_N_HEADS = subnet.SEM_SEG_HEAD.ENC_N_HEADS + [random.choice(self.choices.SEM_SEG_HEAD.ENC_N_HEADS) for _ in range(subnet.SEM_SEG_HEAD.ENC_DEPTHS - old_depth)]
                        subnet.SEM_SEG_HEAD.ENC_N_POINTS = subnet.SEM_SEG_HEAD.ENC_N_POINTS + [random.choice(self.choices.SEM_SEG_HEAD.ENC_N_POINTS) for _ in range(subnet.SEM_SEG_HEAD.ENC_DEPTHS - old_depth)]
                        subnet.SEM_SEG_HEAD.ENC_MLP_RATIO = subnet.SEM_SEG_HEAD.ENC_MLP_RATIO + [random.choice(self.choices.SEM_SEG_HEAD.ENC_MLP_RATIO) for _ in range(subnet.SEM_SEG_HEAD.ENC_DEPTHS - old_depth)]
                    else:
                        subnet.SEM_SEG_HEAD.ENC_N_HEADS = subnet.SEM_SEG_HEAD.ENC_N_HEADS[:subnet.SEM_SEG_HEAD.ENC_DEPTHS]
                        subnet.SEM_SEG_HEAD.ENC_N_POINTS = subnet.SEM_SEG_HEAD.ENC_N_POINTS[:subnet.SEM_SEG_HEAD.ENC_DEPTHS]
                        subnet.SEM_SEG_HEAD.ENC_MLP_RATIO = subnet.SEM_SEG_HEAD.ENC_MLP_RATIO[:subnet.SEM_SEG_HEAD.ENC_DEPTHS]
                # ENC_N_POINTS, ENC_N_HEADS, ENC_MLP_RATIO
                for i in range(subnet.SEM_SEG_HEAD.ENC_DEPTHS):
                    if random.random() < m_prob:
                        subnet.SEM_SEG_HEAD.ENC_N_HEADS[i] = random.choice(self.choices.SEM_SEG_HEAD.ENC_N_HEADS)
                    if random.random() < m_prob:
                        subnet.SEM_SEG_HEAD.ENC_N_POINTS[i] = random.choice(self.choices.SEM_SEG_HEAD.ENC_N_POINTS)
                    if random.random() < m_prob:
                        subnet.SEM_SEG_HEAD.ENC_MLP_RATIO[i] = random.choice(self.choices.SEM_SEG_HEAD.ENC_MLP_RATIO)
                # MASK_DIM
                if random.random() < m_prob:
                    subnet.SEM_SEG_HEAD.MASK_DIM = random.choice(self.choices.SEM_SEG_HEAD.MASK_DIM)

            # MASK_FORMER
            if subnet.MASK_FORMER is not None:
                # DEC_IN_FEATURES
                if random.random() < m_prob:
                    subnet.MASK_FORMER.DEC_IN_FEATURES = random.choice(self.choices.MASK_FORMER.DEC_IN_FEATURES)
                # HIDDEN_DIM
                if random.random() < m_prob:
                    subnet.MASK_FORMER.HIDDEN_DIM = random.choice(self.choices.MASK_FORMER.HIDDEN_DIM)
                # DEC_DEPTHS
                if random.random() < d_prob:
                    old_depth = subnet.MASK_FORMER.DEC_DEPTHS
                    subnet.MASK_FORMER.DEC_DEPTHS = random.choice(self.choices.MASK_FORMER.DEC_DEPTHS)
                    if subnet.MASK_FORMER.DEC_DEPTHS > old_depth:
                        subnet.MASK_FORMER.DEC_CROSS_N_HEADS = subnet.MASK_FORMER.DEC_CROSS_N_HEADS + [random.choice(self.choices.MASK_FORMER.DEC_N_HEADS) for _ in range(subnet.MASK_FORMER.DEC_DEPTHS - old_depth)]
                        subnet.MASK_FORMER.DEC_SELF_N_HEADS = subnet.MASK_FORMER.DEC_SELF_N_HEADS + [random.choice(self.choices.MASK_FORMER.DEC_N_HEADS) for _ in range(subnet.MASK_FORMER.DEC_DEPTHS - old_depth)]
                        subnet.MASK_FORMER.DEC_FFN_MLP_RATIO = subnet.MASK_FORMER.DEC_FFN_MLP_RATIO + [random.choice(self.choices.MASK_FORMER.DEC_MLP_RATIO) for _ in range(subnet.MASK_FORMER.DEC_DEPTHS - old_depth)]
                    else:
                        subnet.MASK_FORMER.DEC_CROSS_N_HEADS = subnet.MASK_FORMER.DEC_CROSS_N_HEADS[:subnet.MASK_FORMER.DEC_DEPTHS-1]
                        subnet.MASK_FORMER.DEC_SELF_N_HEADS = subnet.MASK_FORMER.DEC_SELF_N_HEADS[:subnet.MASK_FORMER.DEC_DEPTHS-1]
                        subnet.MASK_FORMER.DEC_FFN_MLP_RATIO = subnet.MASK_FORMER.DEC_FFN_MLP_RATIO[:subnet.MASK_FORMER.DEC_DEPTHS-1]
                # DEC_CROSS_N_HEADS, DEC_SELF_N_HEADS, DEC_FFN_MLP_RATIO
                for i in range(subnet.MASK_FORMER.DEC_DEPTHS-1):
                    if random.random() < m_prob:
                        subnet.MASK_FORMER.DEC_CROSS_N_HEADS[i] = random.choice(self.choices.MASK_FORMER.DEC_N_HEADS)
                    if random.random() < m_prob:
                        subnet.MASK_FORMER.DEC_SELF_N_HEADS[i] = random.choice(self.choices.MASK_FORMER.DEC_N_HEADS)
                    if random.random() < m_prob:
                        subnet.MASK_FORMER.DEC_FFN_MLP_RATIO[i] = random.choice(self.choices.MASK_FORMER.DEC_MLP_RATIO)
                # MULTI_SCALE_PER_LAYER
                subnet.MASK_FORMER.MULTI_SCALE_PER_LAYER = (subnet.MASK_FORMER.DEC_IN_FEATURES * ((subnet.MASK_FORMER.DEC_DEPTHS-1-1) // len(subnet.MASK_FORMER.DEC_IN_FEATURES) + 1))[:subnet.MASK_FORMER.DEC_DEPTHS-1][::-1]

            return cfg2cand(subnet)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('mutation {}/{}'.format(len(res), mutation_num))

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            subnet1 = cand2cfg(p1)
            subnet2 = cand2cfg(p2)
            max_iters_tmp = 50
            while (subnet1.BACKBONE.DEPTHS != subnet2.BACKBONE.DEPTHS or subnet1.SEM_SEG_HEAD.ENC_DEPTHS != subnet2.SEM_SEG_HEAD.ENC_DEPTHS or subnet1.MASK_FORMER.DEC_DEPTHS != subnet2.MASK_FORMER.DEC_DEPTHS) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
                subnet1 = cand2cfg(p1)
                subnet2 = cand2cfg(p2)

            subnet = CN()

            # BACKBONE
            if subnet1.BACKBONE is not None and subnet2.BACKBONE is not None:
                subnet.BACKBONE = CN()
                subnet.BACKBONE.EMBED_DIM = random.choice([subnet1.BACKBONE.EMBED_DIM, subnet2.BACKBONE.EMBED_DIM])
                subnet.BACKBONE.DEPTHS = [random.choice([i, j]) for i, j in zip(subnet1.BACKBONE.DEPTHS, subnet2.BACKBONE.DEPTHS)]
                subnet.BACKBONE.NUM_HEADS = []
                subnet.BACKBONE.WINDOW_SIZE = []
                subnet.BACKBONE.MLP_RATIO = []
                for i in range(len(subnet.BACKBONE.DEPTHS)):
                    num_heads1 = subnet1.BACKBONE.NUM_HEADS[sum(subnet1.BACKBONE.DEPTHS[:i]):sum(subnet1.BACKBONE.DEPTHS[:i+1])]
                    num_heads2 = subnet2.BACKBONE.NUM_HEADS[sum(subnet2.BACKBONE.DEPTHS[:i]):sum(subnet2.BACKBONE.DEPTHS[:i+1])]
                    num_heads = [random.choice([hi, hj]) for hi, hj in zip(num_heads1, num_heads2)]
                    if len(num_heads) < subnet.BACKBONE.DEPTHS[i]:
                        num_heads = num_heads + num_heads2[subnet1.BACKBONE.DEPTHS[i]:] if subnet1.BACKBONE.DEPTHS[i] < subnet2.BACKBONE.DEPTHS[i] else num_heads1[subnet2.BACKBONE.DEPTHS[i]:]
                    subnet.BACKBONE.NUM_HEADS.extend(num_heads)
                    window_size1 = subnet1.BACKBONE.WINDOW_SIZE[sum(subnet1.BACKBONE.DEPTHS[:i]):sum(subnet1.BACKBONE.DEPTHS[:i+1])]
                    window_size2 = subnet2.BACKBONE.WINDOW_SIZE[sum(subnet2.BACKBONE.DEPTHS[:i]):sum(subnet2.BACKBONE.DEPTHS[:i+1])]
                    window_size = [random.choice([wi, wj]) for wi, wj in zip(window_size1, window_size2)]
                    if len(window_size) < subnet.BACKBONE.DEPTHS[i]:
                        window_size = window_size + window_size2[subnet1.BACKBONE.DEPTHS[i]:] if subnet1.BACKBONE.DEPTHS[i] < subnet2.BACKBONE.DEPTHS[i] else window_size1[subnet2.BACKBONE.DEPTHS[i]:]
                    subnet.BACKBONE.WINDOW_SIZE.extend(window_size)
                    mlp_ratio1 = subnet1.BACKBONE.MLP_RATIO[sum(subnet1.BACKBONE.DEPTHS[:i]):sum(subnet1.BACKBONE.DEPTHS[:i+1])]
                    mlp_ratio2 = subnet2.BACKBONE.MLP_RATIO[sum(subnet2.BACKBONE.DEPTHS[:i]):sum(subnet2.BACKBONE.DEPTHS[:i+1])]
                    mlp_ratio = [random.choice([ri, rj]) for ri, rj in zip(mlp_ratio1, mlp_ratio2)]
                    if len(mlp_ratio) < subnet.BACKBONE.DEPTHS[i]:
                        mlp_ratio = mlp_ratio + mlp_ratio2[subnet1.BACKBONE.DEPTHS[i]:] if subnet1.BACKBONE.DEPTHS[i] < subnet2.BACKBONE.DEPTHS[i] else mlp_ratio1[subnet2.BACKBONE.DEPTHS[i]:]
                    subnet.BACKBONE.MLP_RATIO.extend(mlp_ratio)
            else:
                subnet.BACKBONE = None

            # SEM_SEG_HEAD
            if subnet1.SEM_SEG_HEAD is not None and subnet2.SEM_SEG_HEAD is not None:
                subnet.SEM_SEG_HEAD = CN()
                subnet.SEM_SEG_HEAD.IN_FEATURES = random.choice([subnet1.SEM_SEG_HEAD.IN_FEATURES, subnet2.SEM_SEG_HEAD.IN_FEATURES])
                subnet.SEM_SEG_HEAD.ENC_IN_FEATURES = random.choice([subnet1.SEM_SEG_HEAD.ENC_IN_FEATURES, subnet2.SEM_SEG_HEAD.ENC_IN_FEATURES])
                subnet.SEM_SEG_HEAD.CONVS_DIM = random.choice([subnet1.SEM_SEG_HEAD.CONVS_DIM, subnet2.SEM_SEG_HEAD.CONVS_DIM])
                subnet.SEM_SEG_HEAD.ENC_DEPTHS = random.choice([subnet1.SEM_SEG_HEAD.ENC_DEPTHS, subnet2.SEM_SEG_HEAD.ENC_DEPTHS])
                enc_n_heads1 = subnet1.SEM_SEG_HEAD.ENC_N_HEADS
                enc_n_heads2 = subnet2.SEM_SEG_HEAD.ENC_N_HEADS
                enc_n_heads = [random.choice([hi, hj]) for hi, hj in zip(enc_n_heads1, enc_n_heads2)]
                if len(enc_n_heads) < subnet.SEM_SEG_HEAD.ENC_DEPTHS:
                    enc_n_heads = enc_n_heads + enc_n_heads2[subnet1.SEM_SEG_HEAD.ENC_DEPTHS:] if subnet1.SEM_SEG_HEAD.ENC_DEPTHS < subnet2.SEM_SEG_HEAD.ENC_DEPTHS else enc_n_heads1[subnet2.SEM_SEG_HEAD.ENC_DEPTHS:]
                subnet.SEM_SEG_HEAD.ENC_N_HEADS = enc_n_heads
                enc_n_points1 = subnet1.SEM_SEG_HEAD.ENC_N_POINTS
                enc_n_points2 = subnet2.SEM_SEG_HEAD.ENC_N_POINTS
                enc_n_points = [random.choice([pi, pj]) for pi, pj in zip(enc_n_points1, enc_n_points2)]
                if len(enc_n_points) < subnet.SEM_SEG_HEAD.ENC_DEPTHS:
                    enc_n_points = enc_n_points + enc_n_points2[subnet1.SEM_SEG_HEAD.ENC_DEPTHS:] if subnet1.SEM_SEG_HEAD.ENC_DEPTHS < subnet2.SEM_SEG_HEAD.ENC_DEPTHS else enc_n_points1[subnet2.SEM_SEG_HEAD.ENC_DEPTHS:]
                subnet.SEM_SEG_HEAD.ENC_N_POINTS = enc_n_points
                enc_mlp_ratio1 = subnet1.SEM_SEG_HEAD.ENC_MLP_RATIO
                enc_mlp_ratio2 = subnet2.SEM_SEG_HEAD.ENC_MLP_RATIO
                enc_mlp_ratio = [random.choice([ri, rj]) for ri, rj in zip(enc_mlp_ratio1, enc_mlp_ratio2)]
                if len(enc_mlp_ratio) < subnet.SEM_SEG_HEAD.ENC_DEPTHS:
                    enc_mlp_ratio = enc_mlp_ratio + enc_mlp_ratio2[subnet1.SEM_SEG_HEAD.ENC_DEPTHS:] if subnet1.SEM_SEG_HEAD.ENC_DEPTHS < subnet2.SEM_SEG_HEAD.ENC_DEPTHS else enc_mlp_ratio1[subnet2.SEM_SEG_HEAD.ENC_DEPTHS:]
                subnet.SEM_SEG_HEAD.ENC_MLP_RATIO = enc_mlp_ratio
                subnet.SEM_SEG_HEAD.MASK_DIM = random.choice([subnet1.SEM_SEG_HEAD.MASK_DIM, subnet2.SEM_SEG_HEAD.MASK_DIM])
            else:
                subnet.SEM_SEG_HEAD = None

            # MASK_FORMER
            if subnet1.MASK_FORMER is not None and subnet2.MASK_FORMER is not None:
                subnet.MASK_FORMER = CN()
                subnet.MASK_FORMER.DEC_IN_FEATURES = random.choice([subnet1.MASK_FORMER.DEC_IN_FEATURES, subnet2.MASK_FORMER.DEC_IN_FEATURES])
                subnet.MASK_FORMER.HIDDEN_DIM = random.choice([subnet1.MASK_FORMER.HIDDEN_DIM, subnet2.MASK_FORMER.HIDDEN_DIM])
                subnet.MASK_FORMER.DEC_DEPTHS = random.choice([subnet1.MASK_FORMER.DEC_DEPTHS, subnet2.MASK_FORMER.DEC_DEPTHS])
                dec_cross_n_heads1 = subnet1.MASK_FORMER.DEC_CROSS_N_HEADS
                dec_cross_n_heads2 = subnet2.MASK_FORMER.DEC_CROSS_N_HEADS
                dec_cross_n_heads = [random.choice([chi, chj]) for chi, chj in zip(dec_cross_n_heads1, dec_cross_n_heads2)]
                if len(dec_cross_n_heads) < subnet.MASK_FORMER.DEC_DEPTHS-1:
                    dec_cross_n_heads = dec_cross_n_heads + dec_cross_n_heads2[subnet1.MASK_FORMER.DEC_DEPTHS-1:] if subnet1.MASK_FORMER.DEC_DEPTHS < subnet2.MASK_FORMER.DEC_DEPTHS else dec_cross_n_heads1[subnet2.MASK_FORMER.DEC_DEPTHS-1:]
                subnet.MASK_FORMER.DEC_CROSS_N_HEADS = dec_cross_n_heads
                dec_self_n_heads1 = subnet1.MASK_FORMER.DEC_SELF_N_HEADS
                dec_self_n_heads2 = subnet2.MASK_FORMER.DEC_SELF_N_HEADS
                dec_self_n_heads = [random.choice([shi, shj]) for shi, shj in zip(dec_self_n_heads1, dec_self_n_heads2)]
                if len(dec_self_n_heads) < subnet.MASK_FORMER.DEC_DEPTHS-1:
                    dec_self_n_heads = dec_self_n_heads + dec_self_n_heads2[subnet1.MASK_FORMER.DEC_DEPTHS-1:] if subnet1.MASK_FORMER.DEC_DEPTHS < subnet2.MASK_FORMER.DEC_DEPTHS else dec_self_n_heads1[subnet2.MASK_FORMER.DEC_DEPTHS-1:]
                subnet.MASK_FORMER.DEC_SELF_N_HEADS = dec_self_n_heads
                dec_ffn_mlp_ratio1 = subnet1.MASK_FORMER.DEC_FFN_MLP_RATIO
                dec_ffn_mlp_ratio2 = subnet2.MASK_FORMER.DEC_FFN_MLP_RATIO
                dec_ffn_mlp_ratio = [random.choice([fri, frj]) for fri, frj in zip(dec_ffn_mlp_ratio1, dec_ffn_mlp_ratio2)]
                if len(dec_ffn_mlp_ratio) < subnet.MASK_FORMER.DEC_DEPTHS-1:
                    dec_ffn_mlp_ratio = dec_ffn_mlp_ratio + dec_ffn_mlp_ratio2[subnet1.MASK_FORMER.DEC_DEPTHS-1:] if subnet1.MASK_FORMER.DEC_DEPTHS < subnet2.MASK_FORMER.DEC_DEPTHS else dec_ffn_mlp_ratio1[subnet2.MASK_FORMER.DEC_DEPTHS-1:]
                subnet.MASK_FORMER.DEC_FFN_MLP_RATIO = dec_ffn_mlp_ratio
                subnet.MASK_FORMER.MULTI_SCALE_PER_LAYER = (subnet.MASK_FORMER.DEC_IN_FEATURES * ((subnet.MASK_FORMER.DEC_DEPTHS-1-1) // len(subnet.MASK_FORMER.DEC_IN_FEATURES) + 1))[:subnet.MASK_FORMER.DEC_DEPTHS-1][::-1]
            else:
                subnet.MASK_FORMER = None

            return cfg2cand(subnet)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info('crossover {}/{}'.format(len(res), crossover_num))

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        self.logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            self.logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                self.logger.info('No.{} {} val PQ+AP+mIoU = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.d_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()


@torch.no_grad()
def validate(cfg, data_loader, model, subnet=None):
    logger = logging.getLogger('mask2former')
    model.eval()

    if not isinstance(data_loader, dict):
        data_loader = {'coco_2017_val_panoptic_with_sem_seg': data_loader}

    results = OrderedDict()
    for idx, dataset_name in enumerate(data_loader):
        try:
            evaluator = Trainer.build_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                "or implement its `build_evaluator` method."
            )
            results[dataset_name] = {}
            continue

        results_i = inference_on_dataset(model, data_loader[dataset_name], evaluator, cfg.MODE, subnet, cfg.SEARCH_SPACE)
        results_i = all_gather(results_i)
        results_i = [res for res in results_i if len(res) > 0]
        assert len(results_i) == 1
        results_i = results_i[0]
        results[dataset_name] = results_i
        assert isinstance(
            results_i, dict
        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            results_i
        )
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_super_config(cfg)
    add_search_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    logger = logging.getLogger('mask2former')

    data_loader_val_dict = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader_val_dict[dataset_name] = Trainer.build_test_loader(cfg, dataset_name)

    logger.info(f"Creating SuperModel")
    model = Trainer.build_model(cfg)
    logger.info(str(model))
    model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=('Super' in cfg.MODEL.BACKBONE.NAME+cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME+cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME))

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

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

    t = time.time()
    searcher = EvolutionSearcher(cfg, model, data_loader_val_dict, logger)

    searcher.search()

    logger.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
