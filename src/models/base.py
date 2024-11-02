#!/usr/bin/env python
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch import Tensor

from utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel


class BaseModel(Module, ABC):
    @abstractmethod
    def forward(self, *x: Tensor) -> Tensor:
        """
        See forward of torch.nn.Module
        """
        raise NotImplementedError

    def n_parameters_grad(self) -> int:
        return sum(_.numel() for _ in self.parameters() if _.requires_grad)

    def n_parameters(self) -> int:
        return sum(_.numel() for _ in self.parameters())

    @property
    def data_type(self) -> int:
        return list(self.parameters())[0].dtype

    def load(self, load_model_fn: str):
        state_dict = torch.load(load_model_fn, map_location='cpu')['model']
        if not is_model_dataparallel(self):
            state_dict = rmv_module_dataparallel(state_dict)
        self.load_state_dict(state_dict, strict=True)
