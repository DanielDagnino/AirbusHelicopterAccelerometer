#!/usr/bin/env python
import gc
import inspect
import json
import logging
import random
from typing import Tuple
from path import Path

import numpy as np
import pandas as pd
import torch
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
                 weight: bool = False,
                 transf_degree: float = 0,
                 norm_max_min: bool = True,
                 rank: int = 0,
                 verbose: bool = False
                 ):
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

        # Normalization preprocessing
        if norm_path is not None:
            norm_path = Path(norm_path).expanduser()
            preprocessor = json.load(open(norm_path))
        else:
            preprocessor = dict(mean=0, std=1)

        # Transformations
        if stage == "train":
            self.transforms = build_transforms(preprocessor, transf_degree)
        elif "test" in stage:
            self.transforms = build_transforms_val(preprocessor, transf_degree)
        elif "valid" in stage:
            self.transforms = build_transforms_val(preprocessor, transf_degree)

        self.logger.info(f'{self.__class__.__qualname__} initiated')
        gc.collect()

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.matrix_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        features = np.load(self.matrix_paths[idx])
        features = self.transforms(features)

        if self.weight:
            features = features * (-0.05 / (1 + np.arange(64))).reshape(1, 64)

        if self.norm_max_min:
            features = (features - features.min()) / (features.max() - features.min() + 1.e-8)
            features = 2 * features - 1
        features = torch.tensor(features, dtype=torch.float)

        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = -1

        return features, labels

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        s_features, s_labels = [], []
        for features, labels in batch:
            s_features.append(features)
            s_labels.append(labels)
        s_features = torch.stack(s_features, dim=0)
        s_labels = torch.tensor(s_labels)
        return s_features, s_labels


def main_worker_for_test(this_gpu, n_gpus_per_node):
    setup_logging()

    _N_EXAMPLES = 20
    _COMPUTE_STATS = True

    compute_normalization = True
    for hyperparameters in ['127_126_8']:
        ds = DatasetAirbusHelicopterAccelerometer(
            stage='train',
            dir_matrix=f'~/MyTmp/AirbusHelicopterAccelerometer/data-db/dftrain_127_126_8-1677-120',
            norm_path=None if compute_normalization else '~/MyTmp/AirbusHelicopterAccelerometer/data-db/normalization_127_126_8.json',
            # labels_path='../../data/dfvalid_groundtruth.csv',
            norm_max_min=not compute_normalization,
            transf_degree=0,
        )

        print("dataset")
        for _idx, (_features, _lbls) in enumerate(ds):
            # print(_features.shape, _lbls)
            if _idx + 1 >= _N_EXAMPLES:
                break

        print()
        print("loader")
        loader = DataLoader(
            ds, batch_size=100, num_workers=24,
            pin_memory=True, drop_last=True, shuffle=False, sampler=None, collate_fn=ds.collate_fn
        )
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
