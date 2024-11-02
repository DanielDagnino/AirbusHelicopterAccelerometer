import inspect
import logging
import math
from typing import Union

import numpy as np
from torch.nn import Module

from losses_metrics import losses, metrics
from losses_metrics.metrics import BaseMetric


def get_loss(name: str, kwargs=None, rank=0) -> Module:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if kwargs is None:
        kwargs = dict()

    losses_available = [
        losses.L2,
    ]

    for loss_available in losses_available:
        if name == loss_available.__name__:
            return loss_available(**kwargs)

    msg = f'Wrong loss name {name}.'
    logger.error(msg)
    raise ValueError(msg)


def get_metric(name: str, kwargs=None) -> BaseMetric:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if kwargs is None:
        kwargs = dict()

    metrics_available = [
        metrics.L2,
        metrics.AUC,
    ]

    for metrics_availabl in metrics_available:
        if name == metrics_availabl.__name__:
            return metrics_availabl(**kwargs)

    msg = f'Wrong metric name {name}.'
    logger.error(msg)
    raise ValueError(msg)


class AverageMeter:
    """Stores and computes  the average and current value"""

    def __init__(self, accept_zero_samples: bool = False) -> None:
        self.val: float = math.inf
        self.avg: float = 0.
        self.sum: float = 0.
        self.count: int = 0
        self.accept_zero_samples = accept_zero_samples

    def reset(self) -> None:
        self.val: float = math.inf
        self.avg: float = 0.
        self.sum: float = 0.
        self.count: int = 0

    def val_mean(self) -> float:
        if isinstance(self.val, np.ndarray):
            return float(self.val.mean())
        else:
            return float(self.val)

    def avg_mean(self) -> float:
        if isinstance(self.avg, np.ndarray):
            return float(self.avg.mean())
        else:
            return float(self.avg)

    def update(self, value: Union[float, np.ndarray], batch_size: int = 1) -> None:
        logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        if batch_size == 0 and not self.accept_zero_samples:
            msg = 'Zero values passed are not allowed to compute a mean value.'
            logger.error(msg)
            raise ValueError(msg)

        if batch_size != 0:
            self.val = value
            self.sum += value * batch_size
            self.count += batch_size
            self.avg = self.sum / (self.count + 1.e-6)
