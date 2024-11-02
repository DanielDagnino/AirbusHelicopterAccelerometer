from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn import Module

from losses_metrics import AverageMeter


def reduce_average_meter(average_meter: AverageMeter, this_gpu):
    metrics_sum_all = torch.tensor(average_meter.sum).cuda(this_gpu)
    metrics_cnt_all = torch.tensor(average_meter.count).cuda(this_gpu)
    dist.all_reduce(metrics_sum_all, op=ReduceOp.SUM)
    dist.all_reduce(metrics_cnt_all, op=ReduceOp.SUM)
    average_meter.avg = (metrics_sum_all / metrics_cnt_all).cpu().numpy()
    average_meter.sum = metrics_sum_all.cpu().numpy()
    average_meter.count = metrics_cnt_all.cpu().numpy()


def is_state_dict_dataparallel(state_dict):
    return all([_[:7] == 'module.' for _ in state_dict.keys()])


def is_model_dataparallel(model: Module):
    return all([_[:7] == 'module.' for _ in model.state_dict().keys()])


def naked_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def add_module_dataparallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[f'module.{k}'] = v
    return new_state_dict


def rmv_module_dataparallel(state_dict):
    if is_state_dict_dataparallel(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        return new_state_dict
    else:
        return state_dict
