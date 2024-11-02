import inspect
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor


class BaseMetric(ABC):
    def __init__(self, name_tgt=None):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        self.alias = self.__class__.__name__
        if name_tgt is not None:
            self.alias += f'-{name_tgt}'
        self.name_tgt = name_tgt

    @abstractmethod
    def __call__(self, logits: Tensor, tgts: Tensor, true_labels: Tensor) -> float:
        raise NotImplementedError


class L2(BaseMetric):
    def __init__(self, name_tgt=None):
        super().__init__(name_tgt)

    def __call__(self, logits: Tensor, tgts: Tensor, true_labels: Tensor) -> float:
        error = torch.pow(logits - tgts, 2)
        error = error.mean()
        return error.cpu().item()


class AUC(BaseMetric):
    def __init__(self, name_tgt=None):
        super().__init__(name_tgt)

    def __call__(self, logits: Tensor, tgts: Tensor, true_labels: Tensor) -> float:
        logits = logits.cpu().numpy()
        tgts = tgts.cpu().numpy()
        true_labels = true_labels.cpu().numpy()
        reconstruction_error = np.mean((logits - tgts) ** 2, axis=(1, 2))
        error = roc_auc_score(true_labels, reconstruction_error)
        return error
