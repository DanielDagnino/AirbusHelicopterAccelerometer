#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn
import yaml
from easydict import EasyDict
from path import Path
from tqdm import tqdm

from cli.eval_utils import plot_eval
from dataset.build_transf import build_transforms_val
from models.helpers import load_checkpoint
from models.model_zoo import Feature2DAirbusHelicopterAccelerometer
from utils.general.custom_yaml import init_custom_yaml
from utils.general.modifier import dict_modifier


def run(cfg, percentile, n_max=None):
    cudnn.benchmark = True

    n_feat = cfg.engine.model.args.input_dim

    model = Feature2DAirbusHelicopterAccelerometer(**cfg.engine.model.args)
    model.cuda()
    _, _, _, _ = load_checkpoint(model, cfg.engine.model.resume.load_model_fn, None, None, None, torch.device("cuda"),
                                 False, False, True)
    model.eval()

    if cfg.dataset.valid.norm_path is not None:
        cfg.dataset.valid.norm_path = Path(cfg.dataset.valid.norm_path).expanduser()
    preprocessor = json.load(open(cfg.dataset.valid.norm_path)) if cfg.dataset.valid.norm_path is not None else dict(
        mean=0, std=1)
    transforms = build_transforms_val(preprocessor, transf_degree=0)

    # Read and format data
    cfg.dataset.valid.df_path = Path(cfg.dataset.valid.df_path).expanduser()
    n_wind = int(Path(cfg.dataset.valid.df_path).stem.split('-')[-1])
    n_samples = int(Path(cfg.dataset.valid.df_path).stem.split('-')[-2])
    df = np.load(cfg.dataset.valid.df_path)
    df = df.reshape(n_samples, n_wind, n_feat, n_feat)[:n_max]
    df_gt = pd.read_csv(cfg.dataset.valid.labels_path)[:n_max]
    assert len(df) == len(df_gt)

    # Loop over samples.
    dir_out = Path(cfg.engine.model.resume.load_model_fn).expanduser().parent.parent
    dir_out = dir_out / 'val'
    dir_out.makedirs_p()
    if True or not (Path(dir_out) / 'results_eval.csv').exists():
        with torch.no_grad():
            for isample in tqdm(range(n_samples)):
                features = df[isample]
                reconstruction_errors = []
                for freqwind_features in features:
                    if cfg.dataset.valid.weight == "pow1":
                        freqwind_features = freqwind_features * (-0.05 / (1 + np.arange(64))).reshape(1, 64)
                    elif cfg.dataset.valid.weight == "pow2":
                        freqwind_features = freqwind_features * np.power(-0.05 / (1 + np.arange(64)), 2).reshape(1, 64)
                    elif cfg.dataset.valid.weight != "pow0":
                        raise ValueError(f'Not a valid weight = {cfg.dataset.valid.weight}')

                    freqwind_features = transforms(freqwind_features)

                    if cfg.dataset.valid.norm_max_min:
                        frq_min = freqwind_features.min()
                        frq_max = freqwind_features.max()
                        freqwind_features = (freqwind_features - frq_min) / (frq_max - frq_min + 1.e-8)
                        freqwind_features = 2 * freqwind_features - 1

                    freqwind_features = torch.tensor(freqwind_features, dtype=torch.float)
                    freqwind_features = freqwind_features.unsqueeze(0)
                    freqwind_features = freqwind_features.cuda(non_blocking=True)

                    logits, _, _ = model(freqwind_features, is_train=False)

                    freqwind_features = freqwind_features.squeeze(0)
                    logits = logits.squeeze(0)
                    logits = logits.cpu().numpy()
                    freqwind_features = freqwind_features.cpu().numpy()

                    # Reduction from 2d-matrix to single frame error
                    l2 = np.power(logits - freqwind_features, 2)
                    mask = l2 > percentile   # 10 * pow2 np.percentile(l2, 99) on training set
                    # reconstruction_error = mask * l2
                    reconstruction_error = mask
                    # reconstruction_error = l2

                    # reconstruction_error = reconstruction_error.max().item()
                    reconstruction_error = reconstruction_error.mean().item()

                    reconstruction_errors.append(reconstruction_error)

                # Reduction from N-essim temporal window to a single error
                # df_gt.loc[isample, 'reconstruction_error'] = np.max(reconstruction_errors)
                # df_gt.loc[isample, 'reconstruction_error'] = np.mean(reconstruction_errors)
                df_gt.loc[isample, 'reconstruction_error'] = np.log(np.mean(reconstruction_errors) + 1.e-9)

                if n_max is not None and isample == n_max-1:
                    break

        df_gt.to_csv(Path(dir_out) / 'results_eval.csv', index=False)
    else:
        df_gt = pd.read_csv(Path(dir_out) / 'results_eval.csv')

    plot_eval(df_gt, dir_out)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a model for anomaly detection using Airbus Helicopter Accelerometer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "cfg_fn", type=Path,
        help="Path to the configuration file specifying model and training parameters."
    )
    parser.add_argument(
        "encoder_name", type=str,
        help="Name of the model encoder to be used (e.g., 'resnet50')."
    )
    parser.add_argument(
        "--percentile", type=float, default=0.0015,
        help="Threshold value to filter-out errors. Errors below that will be set to zero."
    )
    parser.add_argument(
        "--n_max", type=int, default=None,
        help="Optional maximum number of samples to be processed (useful for quick testing)."
    )
    parser.add_argument(
        "--model_ckpt", type=str, default=None,
        help="Checkpoint file path to load a pre-trained model. Overrides the default checkpoint in the configuration "
             "file if provided."
    )
    args = parser.parse_args(args)

    args.cfg_fn = Path(args.cfg_fn).expanduser()

    # Configuration.
    init_custom_yaml()
    cfg = yaml.load(open(args.cfg_fn), Loader=yaml.Loader)
    cfg = dict_modifier(config=cfg, modifiers="modifiers",
                        pre_modifiers={"HOME": os.path.expanduser("~"), "ENCODER_NAME": args.encoder_name})
    cfg = EasyDict(cfg)

    if args.model_ckpt is not None:
        args.model_ckpt = Path(args.model_ckpt).expanduser()
        cfg.engine.model.resume.load_model_fn = args.model_ckpt

    run(cfg, percentile=args.percentile, n_max=args.n_max)


if __name__ == "__main__":
    main(sys.argv[1:])
