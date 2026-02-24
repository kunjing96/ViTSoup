import random
from re import subn

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import os
import json
from timm.utils.model import unwrap_model
from timm.utils import accuracy, AverageMeter
from yacs.config import CfgNode as CN

from config import get_config
from logger import create_logger
from data import build_loader
from models import build_model
from utils import load_pretrained, reduce_tensor
from main import sample_configs, throughput


def cfg2cand(cfg):
    cand = []
    cand.append(cfg.EMBED_DIM)
    cand.append(tuple(cfg.DEPTHS))
    cand.append(tuple(cfg.NUM_HEADS))
    cand.append(tuple(cfg.WINDOW_SIZE))
    cand.append(tuple(cfg.MLP_RATIO))
    return tuple(cand)


def cand2cfg(cand):
    cfg = CN()
    cfg.EMBED_DIM = cand[0]
    cfg.DEPTHS = list(cand[1])
    cfg.NUM_HEADS = list(cand[2])
    cfg.WINDOW_SIZE = list(cand[3])
    cfg.MLP_RATIO = list(cand[4])
    return cfg


class EvolutionSearcher(object):

    def __init__(self, config, model, model_without_ddp, val_loader, test_loader):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.config = config
        self.max_epochs = config.SEARCH.MAX_EPOCHS
        self.select_num = config.SEARCH.SELECT_NUM
        self.population_num = config.SEARCH.POPULATION_NUM
        self.m_prob = config.SEARCH.M_PROB
        self.d_prob =config.SEARCH.D_PROB
        self.crossover_num = config.SEARCH.CROSSOVER_NUM
        self.mutation_num = config.SEARCH.MUTATION_NUM
        self.parameters_limits = config.SEARCH.PARAM_LIMITS
        self.min_parameters_limits = config.SEARCH.MIN_PARAM_LIMITS
        self.flops_limits = config.SEARCH.FLOPS_LIMITS
        self.min_flops_limits = config.SEARCH.MIN_FLOPS_LIMITS
        self.throughput_limits = config.SEARCH.THROUGHPUT_LIMITS
        self.min_throughput_limits = config.SEARCH.MIN_THROUGHPUT_LIMITS
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = config.OUTPUT
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = config.SEARCH.RESUME
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = config.SUPER

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
        logger.info('save checkpoint to {}'.format(checkpoint_path))

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
        logger.info('load checkpoint from {}', (self.checkpoint_path))
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
        flops = model_module.flops()
        _throughput, _ = throughput(self.val_loader, model_module, None)
        info['params'] =  n_parameters / 10.**6
        info['flops'] =  flops / 10.**9
        info['throughput'] =  _throughput

        if self.parameters_limits is not None and info['params'] > self.parameters_limits:
            logger.info('parameters limit exceed')
            return False

        if self.min_parameters_limits is not None and info['params'] < self.min_parameters_limits:
            logger.info('under minimum parameters limit')
            return False

        if self.flops_limits is not None and info['flops'] > self.flops_limits:
            logger.info('flops limit exceed')
            return False

        if self.min_flops_limits is not None and info['flops'] < self.min_flops_limits:
            logger.info('under minimum flops limit')
            return False

        if self.throughput_limits is not None and info['throughput'] > self.throughput_limits:
            logger.info('throughput limit exceed')
            return False

        if self.min_throughput_limits is not None and info['throughput'] < self.min_throughput_limits:
            logger.info('under minimum throughput limit')
            return False

        logger.info("rank: {} {} {} {} {}".format(dist.get_rank(), subnet, info['params'], info['flops'], info['throughput']))
        val_acc1, val_acc5, val_loss = validate(config, self.val_loader, self.model, subnet=subnet)
        test_acc1, test_acc5, test_loss = validate(config, self.test_loader, self.model, subnet=subnet)

        info['acc'] = val_acc1
        info['test_acc'] = test_acc1

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        logger.info('select ......')
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
        logger.info('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))
        logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, d_prob):
        assert k in self.keep_top_k
        logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            subnet = cand2cfg(cand)

            # EMBED_DIM
            if random.random() < m_prob:
                subnet.EMBED_DIM = random.choice(self.choices.EMBED_DIM)

            # DEPTHS
            for i in range(len(subnet.DEPTHS)):
                if random.random() < d_prob:
                    old_depth = subnet.DEPTHS[i]
                    left = sum(subnet.DEPTHS[:i])
                    right = sum(subnet.DEPTHS[:i+1])
                    subnet.DEPTHS[i] = random.choice(self.choices.DEPTHS[i])
                    if subnet.DEPTHS[i] > old_depth:
                        subnet.NUM_HEADS = subnet.NUM_HEADS[:right] + [random.choice(self.choices.NUM_HEADS[i]) for _ in range(subnet.DEPTHS[i] - old_depth)] + subnet.NUM_HEADS[right:]
                        subnet.WINDOW_SIZE = subnet.WINDOW_SIZE[:right] + [random.choice(self.choices.WINDOW_SIZE) for _ in range(subnet.DEPTHS[i] - old_depth)] + subnet.WINDOW_SIZE[right:]
                        subnet.MLP_RATIO = subnet.MLP_RATIO[:right] + [random.choice(self.choices.MLP_RATIO) for _ in range(subnet.DEPTHS[i] - old_depth)] + subnet.MLP_RATIO[right:]
                    else:
                        subnet.NUM_HEADS = subnet.NUM_HEADS[:left] + subnet.NUM_HEADS[left:right][:subnet.DEPTHS[i]] + subnet.NUM_HEADS[right:]
                        subnet.WINDOW_SIZE = subnet.WINDOW_SIZE[:left] + subnet.WINDOW_SIZE[left:right][:subnet.DEPTHS[i]] + subnet.WINDOW_SIZE[right:]
                        subnet.MLP_RATIO = subnet.MLP_RATIO[:left] + subnet.MLP_RATIO[left:right][:subnet.DEPTHS[i]] + subnet.MLP_RATIO[right:]

            # NUM_HEADS, WINDOW_SIZE, MLP_RATIO
            for i in range(len(subnet.DEPTHS)):
                for j in range(subnet.DEPTHS[i]):
                    l = sum(subnet.DEPTHS[:i]) + j
                    if random.random() < m_prob:
                        subnet.NUM_HEADS[l] = random.choice(self.choices.NUM_HEADS[i])
                    if random.random() < m_prob:
                        subnet.WINDOW_SIZE[l] = random.choice(self.choices.WINDOW_SIZE)
                    if random.random() < m_prob:
                        subnet.MLP_RATIO[l] = random.choice(self.choices.MLP_RATIO)

            return cfg2cand(subnet)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('mutation {}/{}'.format(len(res), mutation_num))

        logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            subnet1 = cand2cfg(p1)
            subnet2 = cand2cfg(p2)
            max_iters_tmp = 50
            while subnet1.DEPTHS != subnet2.DEPTHS and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
                subnet1 = cand2cfg(p1)
                subnet2 = cand2cfg(p2)

            subnet = CN()
            subnet.EMBED_DIM = random.choice([subnet1.EMBED_DIM, subnet2.EMBED_DIM])
            subnet.DEPTHS = [random.choice([i, j]) for i, j in zip(subnet1.DEPTHS, subnet2.DEPTHS)]
            subnet.NUM_HEADS = []
            subnet.WINDOW_SIZE = []
            subnet.MLP_RATIO = []
            for i in range(len(subnet.DEPTHS)):
                num_heads1 = subnet1.NUM_HEADS[sum(subnet1.DEPTHS[:i]):sum(subnet1.DEPTHS[:i+1])]
                num_heads2 = subnet2.NUM_HEADS[sum(subnet2.DEPTHS[:i]):sum(subnet2.DEPTHS[:i+1])]
                num_heads = [random.choice([i, j]) for i, j in zip(num_heads1, num_heads2)]
                if len(num_heads) < subnet.DEPTHS[i]:
                    num_heads = num_heads + num_heads2[subnet1.DEPTHS[i]:] if subnet1.DEPTHS[i] < subnet2.DEPTHS[i] else num_heads1[subnet2.DEPTHS[i]:]
                subnet.NUM_HEADS.extend(num_heads)
                window_size1 = subnet1.WINDOW_SIZE[sum(subnet1.DEPTHS[:i]):sum(subnet1.DEPTHS[:i+1])]
                window_size2 = subnet2.WINDOW_SIZE[sum(subnet2.DEPTHS[:i]):sum(subnet2.DEPTHS[:i+1])]
                window_size = [random.choice([i, j]) for i, j in zip(window_size1, window_size2)]
                if len(window_size) < subnet.DEPTHS[i]:
                    window_size = window_size + window_size2[subnet1.DEPTHS[i]:] if subnet1.DEPTHS[i] < subnet2.DEPTHS[i] else window_size1[subnet2.DEPTHS[i]:]
                subnet.WINDOW_SIZE.extend(window_size)
                mlp_ratio1 = subnet1.MLP_RATIO[sum(subnet1.DEPTHS[:i]):sum(subnet1.DEPTHS[:i+1])]
                mlp_ratio2 = subnet2.MLP_RATIO[sum(subnet2.DEPTHS[:i]):sum(subnet2.DEPTHS[:i+1])]
                mlp_ratio = [random.choice([i, j]) for i, j in zip(mlp_ratio1, mlp_ratio2)]
                if len(mlp_ratio) < subnet.DEPTHS[i]:
                    mlp_ratio = mlp_ratio + mlp_ratio2[subnet1.DEPTHS[i]:] if subnet1.DEPTHS[i] < subnet2.DEPTHS[i] else mlp_ratio1[subnet2.DEPTHS[i]:]
                subnet.MLP_RATIO.extend(mlp_ratio)
    
            return cfg2cand(subnet)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('crossover {}/{}'.format(len(res), crossover_num))

        logger.info('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                logger.info('No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['test_acc'], self.vis_dict[cand]['params']))
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
def validate(config, data_loader, model, subnet=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    if 'super' in config.MODEL.TYPE:
        if subnet is None:
            if config.MODE == 'super':
                subnet = sample_configs(config.SUPER)
            else:
                subnet = config.SUBNET
        model_module = unwrap_model(model)
        model_module.set_sample_config(subnet)
        n_parameters = model_module.params()
        flops = model_module.flops()
        _throughput, _ = throughput(data_loader, model_module, None)
        logger.info(f"sampled model config: {subnet}")
        logger.info(f"sampled model params: {n_parameters / 1e6}M")
        logger.info(f"sampled model FLOPs: {flops / 1e9}G")
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
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
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

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--d_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=30.0)
    parser.add_argument('--min-param-limits', type=float, default=0.0)
    parser.add_argument('--flops-limits', type=float, default=5.0)
    parser.add_argument('--min-flops-limits', type=float, default=0.0)
    parser.add_argument('--throughput-limits', type=float, default=None)
    parser.add_argument('--min-throughput-limits', type=float, default=None)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_val, dataset_test, data_loader_val, data_loader_test, mixup_fn = build_loader(config, is_search=True)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    model.cuda()
    model_without_ddp = model

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

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)

    t = time.time()
    searcher = EvolutionSearcher(config, model, model_without_ddp, data_loader_val, data_loader_test)

    searcher.search()

    logger.info('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


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
