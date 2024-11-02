#!/usr/bin/env python
import gc
import inspect
import json
import logging
import time
from typing import Dict, Optional

import torch
from path import Path
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses_metrics import AverageMeter
from losses_metrics.metrics import BaseMetric
from models.helpers import save_checkpoint
from optimizer import RETURN_optimizer_builder
from optimizer.clip import Clipper
from scheduler import RETURN_scheduler_builder
from utils.torch.dataparallel import reduce_average_meter


def trainer(epoch: int,
            data_loader: DataLoader,
            model: Module,
            optimizer: RETURN_optimizer_builder,
            lr_scheduler: RETURN_scheduler_builder,
            step_scheduler_at_save: bool,
            loss_funct: Module,
            metric_functs: Optional[Dict[str, Dict[str, BaseMetric]]],
            scaler: GradScaler = None,
            writer: SummaryWriter = None,
            fn_resume: str = None,
            stage: str = "train",
            clipper: Clipper = None,
            grad_accum: int = 1,
            this_gpu: int = 0,
            rank: int = 0,
            n_log_interval: int = 100,
            n_save_inter_epoch: int = 100,
            save_tmp_model_fn: str = None,
            non_blocking: bool = True,
            distributed_data_parallel: bool = False,
            ) -> (float, Dict[str, float], Module):
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    # Select whether trainable or not.
    if stage == "train":
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # Meters.
    loss_meter = AverageMeter()
    metric_meters = {metric_name: AverageMeter(accept_zero_samples=False)
                     for metric_name in metric_functs[stage].keys()}
    fn_resume = Path(fn_resume)
    resume_it = dict()

    # Loop over mini-batches.
    if rank == 0:
        logger.info(f"{stage} Ep={epoch}")
    it = None
    start_time = time.time()
    for it, (feat, labels) in enumerate(data_loader):
        bs = feat.shape[0]
        feat = feat.detach().requires_grad_(requires_grad=False).cuda(this_gpu, non_blocking=non_blocking)
        labels = labels.detach().requires_grad_(requires_grad=False)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float32 if scaler is None else torch.float16):
                feat_recon, mean, log_var = model(feat)
                loss, recon_loss, kl_loss = loss_funct(feat_recon, feat, mean, log_var)
        else:
            feat_recon, mean, log_var = model(feat)
            loss, recon_loss, kl_loss = loss_funct(feat_recon, feat, mean, log_var)

        if torch.isnan(loss):
            msg = f"NaN loss found at it={it}"
            logger.error(msg)
            raise ValueError(msg)

        if stage == "train":
            loss = (1. / grad_accum) * loss
            if scaler is not None:
                scaler.scale(loss).backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        scaler.unscale_(optimizer)
                        clipper.apply_to_grad(model)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        clipper.apply_to_grad(model)
                    optimizer.step()
                    optimizer.zero_grad()

        loss_meter.update(grad_accum * loss.item(), bs)

        with torch.no_grad():
            for metric_name, metric_funct in metric_functs[stage].items():
                metric = metric_funct(feat_recon, feat, labels)
                metric_meters[metric_name].update(metric, bs)

        # Intermediate results.
        if (it + 1) % n_log_interval == 0:
            time_elapse = 1000 * (time.time() - start_time)
            if rank == 0:
                logger.info(
                    f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f} | "
                    f"{time_elapse:.5}ms cuda:{this_gpu}")
                for metric_name, metric_meter in metric_meters.items():
                    logger.info(
                        f"{metric_name: <9} | MA {metric_meter.val: .3f} | MT {metric_meter.avg: .3f} | CNT {metric_meter.count}")
                    logger.info(45 * '-')
            start_time = time.time()

        if (it + 1) % n_save_inter_epoch == 0:
            if step_scheduler_at_save:
                lr_scheduler.step()
            if rank == 0:
                logger.info(' ************************************************** ')
                logger.info(f' ***** lr = {optimizer.param_groups[0]["lr"]} ***** ')
                logger.info(' ************************************************** ')
                logger.info(f'Saving temporal model {save_tmp_model_fn}')
                save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                                lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)
                save_checkpoint(save_tmp_model_fn[:-5] + f"_it={it}.ckpt", model=model, optimizer=optimizer,
                                scaler=scaler, lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

            gc.collect()
            torch.cuda.memory_reserved()

    # Final results.
    if distributed_data_parallel and torch.cuda.device_count() > 1:
        for metric_name, metric_meter in metric_meters.items():
            reduce_average_meter(metric_meter, this_gpu)
        reduce_average_meter(loss_meter, this_gpu)

    if rank == 0:
        logger.info(f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f}")
        for metric_name, metric_meter in metric_meters.items():
            resume_it[metric_name] = float(metric_meter.avg)
            logger.info(
                f"{metric_name: <9} | MA {metric_meter.val: .3f} | MT {metric_meter.avg: .3f} | CNT {metric_meter.count}")
            logger.info(45 * '-')

    if writer is not None:
        step = it + epoch * len(data_loader)
        writer.add_scalar(f"Loss/{stage}", loss_meter.avg, global_step=step)
        for metric_name, metric_meter in metric_meters.items():
            writer.add_scalar(f"Metric/{metric_name}", metric_meter.avg, global_step=step)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step=step)

    if rank == 0:
        logger.info(f'Saving temporal model {save_tmp_model_fn}')
        save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                        lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

        resume = json.load(open(fn_resume)) if fn_resume.exists() else dict()
        resume.setdefault(stage, []).append(resume_it)
        json.dump(resume, open(fn_resume, 'w'), indent=4)

    result = metric_meters["AUC"].avg if "AUC" in metric_meters else metric_meters["L2"].avg
    return loss_meter.avg, result, model
