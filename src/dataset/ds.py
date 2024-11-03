#!/usr/bin/env python
import gc
import inspect
import json
import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from path import Path
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.build_transf import build_transforms, build_transforms_val
from utils.logger.logger import setup_logging


class DatasetAirbusHelicopterAccelerometer(Dataset):
    def __init__(self,
                 dir_matrix: str,
                 norm_path: str = None,
                 labels_path: str = None,
                 stage: str = 'train',
                 weight: str = 'pow0',
                 transf_degree: float = 0,
                 norm_max_min: bool = True,
                 rank: int = 0,
                 verbose: bool = False
                 ):
        """
        PyTorch Dataset for loading and preprocessing 2D matrices of Airbus helicopter
        accelerometer data, including normalization, augmentation, and optional weighting
        for frequency bands.

        Parameters:
            dir_matrix (str): Directory containing the 2D matrix .npy files. Assumes the directory
                name includes details on windowing and sampling that are extracted during initialization.
            norm_path (str, optional): Path to a JSON file with mean and std values for normalization.
                If None, normalization defaults to mean=0 and std=1.
            labels_path (str, optional): Path to a CSV file containing labels for anomaly detection,
                used to tag each matrix with anomaly or non-anomaly status. Default is None.
            stage (str): One of {"train", "test", "valid"}, determining which transformations to apply.
            weight (str): Specifies weighting for frequency bands: "pow0" (no weighting), "pow1", or "pow2".
            transf_degree (float): Degree of transformation to apply; only applicable when stage="train".
            norm_max_min (bool): Whether to apply min-max normalization to the features after
                transformation. Default is True.
            rank (int): Rank for logging purposes; primarily used in multi-process settings.
            verbose (bool): If True, enables verbose output for debugging and detailed process information.

        Attributes:
            matrix_paths (List[Path]): List of paths to the individual .npy files containing feature matrices.
            labels (np.ndarray): Array of labels indicating anomalies, repeated as needed for each window
                in the dataset.
            transforms (callable): Transformation pipeline applied to each matrix based on the specified stage.

        Methods:
            print(msg):
                Prints the provided message if verbose mode is enabled.

            __len__():
                Returns the number of samples in the dataset.

            __getitem__(idx):
                Loads, transforms, and returns a feature matrix and its label (or -1 if no label)
                for the specified index.

            collate_fn(batch):
                Collates a batch of samples for DataLoader, stacking features and labels.

        Example usage in the __name__ below.
        """
        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.weight = weight
        self.rank = rank
        self.verbose = verbose
        self.norm_max_min = norm_max_min

        # Extract folding parameters.
        dir_matrix = Path(dir_matrix).expanduser()
        n_wind = int(dir_matrix.stem.split('-')[-1])
        n_samples = int(dir_matrix.stem.split('-')[-2])
        self.matrix_paths = list(dir_matrix.walkfiles('*.npy'))

        # Expand label for the N 2D-matrix samples.
        self.labels = None
        if labels_path is not None:
            labels_path = Path(labels_path).expanduser()
            self.labels = np.array(pd.read_csv(labels_path)['anomaly']).reshape(n_samples)
            self.labels = self.labels.repeat(repeats=n_wind)
            assert len(self.labels) == len(self.matrix_paths)

        # Normalization
        if norm_path is not None:
            norm_path = Path(norm_path).expanduser()
            preprocessor = json.load(open(norm_path))
        else:
            preprocessor = dict(mean=0, std=1)

        # Augmentation + Normalization
        if stage == "train":
            self.transforms = build_transforms(preprocessor, transf_degree)
        elif stage in ["test", "valid"]:
            self.transforms = build_transforms_val(preprocessor, transf_degree)
        else:
            raise ValueError(f'Not accepted stage = {stage}')

        self.logger.info(f'{self.__class__.__qualname__} initiated')
        gc.collect()

    def print(self, msg):
        """
        Prints a message if verbose mode is enabled.

        Args:
            msg (str): The message to print.
        """
        if self.verbose:
            print(msg)

    def __len__(self):
        """
        Returns the total number of samples (feature matrices) available in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.matrix_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        """
        Loads and preprocesses a feature matrix and retrieves its corresponding label,
        if available.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, int]: A tuple containing the preprocessed feature matrix (Tensor)
            and the associated label (int), or -1 if no label is available.
        """
        features = np.load(self.matrix_paths[idx])

        # Augmentation + Normalization
        features = self.transforms(features)

        # Weight the frequencies
        if self.weight == "pow1":
            features = features * (-0.05 / (1 + np.arange(64))).reshape(1, 64)
        elif self.weight == "pow2":
            features = features * np.power(1 / (1 + np.arange(64)), 2).reshape(1, 64)
        elif self.weight != "pow0":
            raise ValueError(f'Not a valid weight = {self.weight}')

        # Normalization: min max and then 1, -1.
        if self.norm_max_min:
            features = (features - features.min()) / (features.max() - features.min() + 1.e-8)
            features = 2 * features - 1
        features = torch.tensor(features, dtype=torch.float)

        # GT labels, otherwise -1 (:= unknown)
        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = -1

        return features, labels

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Custom collate function to prepare a batch for DataLoader, stacking features and labels.

        Args:
            batch (List[Tuple[Tensor, int]]): A list of tuples containing features and labels.

        Returns:
            Tuple[Tensor, Tensor]: A tuple with stacked feature tensors and corresponding labels.
        """
        s_features, s_labels = [], []
        for features, labels in batch:
            s_features.append(features)
            s_labels.append(labels)
        s_features = torch.stack(s_features, dim=0)
        s_labels = torch.tensor(s_labels)
        return s_features, s_labels


def main_worker_for_test(this_gpu, n_gpus_per_node):
    """Test function"""
    setup_logging()

    _N_EXAMPLES = 20
    _COMPUTE_STATS = True

    compute_normalization = True
    for hyperparameters in ['127_126_8']:
        ds = DatasetAirbusHelicopterAccelerometer(stage='train',
            dir_matrix=f'~/MyTmp/AirbusHelicopterAccelerometer/data-db/dftrain_127_126_8-1677-120',
            norm_path=None if compute_normalization else '~/MyTmp/AirbusHelicopterAccelerometer/data-db/normalization_127_126_8.json',
            # labels_path='../../data/dfvalid_groundtruth.csv',
            norm_max_min=not compute_normalization, transf_degree=0, )

        print("dataset")
        for _idx, (_features, _lbls) in enumerate(ds):
            # print(_features.shape, _lbls)
            if _idx + 1 >= _N_EXAMPLES:
                break

        print()
        print("loader")
        loader = DataLoader(ds, batch_size=100, num_workers=24, pin_memory=True, drop_last=True, shuffle=False,
            sampler=None, collate_fn=ds.collate_fn)
        cnt, mean, mean2 = 0, 0, 0
        for _it, (_features, _lbls) in tqdm(enumerate(loader), total=len(loader)):
            bs = _features.shape[0]
            n_twind = _features.shape[1]
            cnt += bs * n_twind
            mean += _features.sum((0, 1)).to(torch.float)
            mean2 += _features.pow(2).sum((0, 1)).to(torch.float)
            # print(_features.shape, _lbls.shape)
            if not _COMPUTE_STATS and _it + 1 >= _N_EXAMPLES:
                break

        mean /= cnt
        mean2 /= cnt
        mean = np.array(mean)
        std = np.sqrt(mean2 - mean * mean)
        mean = [float(_) for _ in mean.tolist()]
        std = [float(_) for _ in std.tolist()]
        if compute_normalization:
            fn_norm_out = f'~/MyTmp/AirbusHelicopterAccelerometer/data-db/normalization_{hyperparameters}.json'
            fn_norm_out = Path(fn_norm_out).expanduser()
            json.dump(dict(mean=mean, std=std), open(fn_norm_out, 'w'), indent=1)
        print()
        print(f"cnt = {cnt}")
        print(f"mean = {mean}")
        print(f"std = {std}")


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    main_worker_for_test(None, None)
