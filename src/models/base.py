#!/usr/bin/env python
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch import Tensor

from utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel


class BaseModel(Module, ABC):
    """
    Abstract base class for deep learning models, extending PyTorch's `Module` and implementing
    some common utility functions for model handling, including parameter counting and loading.

    Methods:
        forward(*x: Tensor) -> Tensor:
            Abstract method to be implemented in derived classes, representing the model's forward pass.
            Must be overridden by subclasses.

        n_parameters_grad() -> int:
            Returns the count of all parameters that require gradients, aiding in understanding the model's
            complexity related to trainable parameters.

        n_parameters() -> int:
            Returns the count of all parameters (trainable and non-trainable), providing an overview of
            the model's total parameter count.

        data_type() -> torch.dtype:
            Property that returns the data type of the model's parameters, inferred from the first parameter.
            This is useful for ensuring consistent data types across the model.

        load(load_model_fn: str):
            Loads model weights from a file. The method manages single and multi-GPU models by removing
            any 'module.' prefixes if the model is not wrapped in DataParallel. This helps seamlessly load
            state dictionaries across different training setups.

    Example usage:
        model = DerivedModel()
        model.load("path/to/model_checkpoint.pth")

        print(f"Trainable parameters: {model.n_parameters_grad()}")
        print(f"Total parameters: {model.n_parameters()}")
    """

    @abstractmethod
    def forward(self, *x: Tensor) -> Tensor:
        """
        Abstract method representing the forward pass. Must be implemented in subclasses
        to define the computation of the model.

        Raises:
            NotImplementedError: This exception is raised if the method is not overridden
            in the subclass.
        """
        raise NotImplementedError

    def n_parameters_grad(self) -> int:
        """
        Counts the number of parameters that require gradients, giving insight into the
        number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_parameters(self) -> int:
        """
        Counts the total number of parameters, providing a measure of the model's complexity.

        Returns:
            int: Total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def data_type(self) -> torch.dtype:
        """
        Infers the data type of the model's parameters, useful for ensuring consistency
        when transferring models across devices or precision settings.

        Returns:
            torch.dtype: The data type of the model's parameters.
        """
        return list(self.parameters())[0].dtype

    def load(self, load_model_fn: str):
        """
        Loads a model state dictionary from the specified file, handling both single-GPU
        and multi-GPU checkpoints. If the model was trained in DataParallel mode, removes
        'module.' prefixes from the keys for compatibility.

        Args:
            load_model_fn (str): Path to the model checkpoint file.

        Raises:
            RuntimeError: If the state dictionary keys do not match the model's architecture.
        """
        state_dict = torch.load(load_model_fn, map_location='cpu')['model']
        if not is_model_dataparallel(self):
            state_dict = rmv_module_dataparallel(state_dict)
        self.load_state_dict(state_dict, strict=True)
