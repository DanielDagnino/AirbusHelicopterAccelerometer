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


def run(cfg, n_feat):
    cudnn.benchmark = True

    model = Feature2DAirbusHelicopterAccelerometer(**cfg.engine.model.args)
    model.cuda()
    _, _, _, _ = load_checkpoint(model, cfg.engine.model.resume.load_model_fn, None, None, None, torch.device("cuda"),
                                 False, False, True)
    model.eval()

    preprocessor = json.load(open(cfg.dataset.valid.norm_path)) if cfg.dataset.valid.norm_path is not None else dict(
        mean=0, std=1)
    transforms = build_transforms_val(preprocessor, transf_degree=0)

    # Read and format data
    cfg.dataset.valid.df_path = Path(cfg.dataset.valid.df_path).expanduser()
    n_wind = int(Path(cfg.dataset.valid.df_path).stem.split('-')[-1])
    n_samples = int(Path(cfg.dataset.valid.df_path).stem.split('-')[-2])
    df = np.load(cfg.dataset.valid.df_path)
    df = df.reshape(n_samples, n_wind, n_feat, n_feat)
    df_gt = pd.read_csv(cfg.dataset.valid.labels_path)
    assert len(df) == len(df_gt)

    # Loop over samples.
    dir_out = Path(cfg.engine.model.resume.load_model_fn).expanduser().parent.parent
    dir_out = dir_out / 'val'
    dir_out.makedirs_p()
    if not (Path(dir_out) / 'results_eval.csv').exists():
        with torch.no_grad():
            for isample in tqdm(range(n_samples)):
                features = df[isample]
                reconstruction_errors = []
                for freqwind_features in features:
                    freqwind_features = transforms(freqwind_features)

                    if cfg.dataset.valid.norm_max_min:
                        frq_min = freqwind_features.min()
                        frq_max = freqwind_features.max()
                        freqwind_features = (freqwind_features - frq_min) / (frq_max - frq_min + 1.e-8)
                        freqwind_features = 2 * freqwind_features - 1

                    freqwind_features = torch.tensor(freqwind_features, dtype=torch.float)
                    freqwind_features = freqwind_features.unsqueeze(0)
                    freqwind_features = freqwind_features.detach().requires_grad_(requires_grad=False).cuda(
                        non_blocking=True)

                    logits, _, _ = model(freqwind_features, is_train=False)

                    freqwind_features = freqwind_features.squeeze(0)
                    logits = logits.squeeze(0)
                    logits = logits.cpu().numpy()
                    freqwind_features = freqwind_features.cpu().numpy()

                    reconstruction_error = np.power(logits - freqwind_features, 2)
                    reconstruction_error = reconstruction_error.max().item()
                    reconstruction_errors.append(reconstruction_error)

                df_gt.loc[isample, 'reconstruction_error'] = np.mean(reconstruction_errors)

        df_gt.to_csv(Path(dir_out) / 'results_eval.csv', index=False)
    else:
        df_gt = pd.read_csv(Path(dir_out) / 'results_eval.csv')

    plot_eval(df_gt, dir_out)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cfg_fn", type=Path, help="Configuration file")
    parser.add_argument("encoder_name", type=str, help="Model encoder")
    parser.add_argument("n_feat", type=int, help="Model encoder")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="If defined, the default ckpt of the cfg file will be changed.")
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

    run(cfg, args.n_feat)


if __name__ == "__main__":
    main(sys.argv[1:])
