import gc
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

from train.trainer import exists, noop, cycle, accum_log
from train.dataset import RexGroundingCTDataset


def custom_collate_fn(batch):
    """
    Collate function for RexGroundingCT samples.
    Each sample is (video_tensor, mask_tensor, embed_dict).

    Args:
        batch: list of tuples, each from RexGroundingCTDataset.__getitem__

    Returns:
        batch_video: (B, 1, D, H, W) stacked images
        batch_masks: (B, 1, D, H, W) stacked masks
        batch_embed_dict: dict with padded text_emb/text_tokens/text_mask
    """
    video_tensors, mask_tensors, embed_dicts = zip(*batch)

    batch_video = torch.stack(video_tensors, dim=0)
    batch_masks = torch.stack(mask_tensors, dim=0)

    text_emb_list = [item["text_emb"] for item in embed_dicts]
    text_tokens_list = [item["text_tokens"] for item in embed_dicts]
    text_mask_list = [item["text_mask"] for item in embed_dicts]
    max_len = max(tok.shape[0] for tok in text_tokens_list)

    padded_emb = []
    padded_tokens = []
    padded_mask = []

    for emb, tok, msk in zip(text_emb_list, text_tokens_list, text_mask_list):
        L = tok.shape[0]
        pad_len = max_len - L
        pad_tok = torch.cat([tok, torch.zeros((pad_len, tok.size(1)), dtype=tok.dtype)])
        pad_msk = torch.cat([msk, torch.zeros(pad_len, dtype=msk.dtype)])
        padded_emb.append(emb)
        padded_tokens.append(pad_tok)
        padded_mask.append(pad_msk)

    batch_embed_dict = {
        "text_emb": torch.stack(padded_emb),
        "text_tokens": torch.stack(padded_tokens),
        "text_mask": torch.stack(padded_mask),
    }

    return batch_video, batch_masks, batch_embed_dict


class RexSegTrainer(nn.Module):
    """
    Trainer class for ReXGroundingCT text-guided segmentation.

    Args:
        seg_model: RexGroundingCTSeg instance (the segmentation model to train).

        --- Dataset configuration (required, forwarded directly to RexGroundingCTDataset) ---
        dataset_center_csv:        path to lung center CSV.
        dataset_json_path:         path to MICCAI_challenge_dataset.json split file.
        dataset_image_dir:         directory containing CT .nii.gz volumes.
        dataset_mask_dir:          directory with per-finding segmentation masks.
        dataset_embed_dir:         directory with pre-computed .pt text embeddings.
        dataset_split:             split key to use (default "train").

        --- Optimization / scheduling ---
        batch_size:                per-GPU batch size for DataLoader.
        num_train_steps:           total number of gradient updates.
        lr:                        AdamW initial learning rate.
        wd:                        AdamW weight decay (kept for API, not used yet).
        max_grad_norm:             gradient clipping threshold (None disables).

        --- Persistence ---
        results_folder:            directory for checkpoints and loss CSVs.
        save_results_every:        step interval for saving loss CSV summaries.
        save_model_every:          step interval for saving model checkpoints.

        --- Infrastructure ---
        num_workers:               DataLoader worker processes.
        accelerate_kwargs:         extra kwargs forwarded to Accelerator.
    """
    def __init__(
        self,
        seg_model,
        *,
        # --- Dataset paths (required, keyword-only) ---
        dataset_center_csv,
        dataset_json_path,
        dataset_image_dir,
        dataset_mask_dir,
        dataset_embed_dir,
        dataset_split="train",
        # --- Optimization ---
        num_train_steps,
        batch_size,
        lr=1.25e-5,
        wd=0.,
        max_grad_norm=0.5,
        # --- Persistence ---
        save_results_every=1000,
        save_model_every=1000,
        results_folder='/results',
        num_workers=8,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        if self.accelerator.state.deepspeed_plugin is not None:
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size

        self.seg_model = seg_model

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        self.max_grad_norm = max_grad_norm
        self.lr = lr

        self.ds = RexGroundingCTDataset(
            center_csv=dataset_center_csv,
            dataset_json_path=dataset_json_path,
            image_dir=dataset_image_dir,
            mask_dir=dataset_mask_dir,
            embed_dir=dataset_embed_dir,
            split=dataset_split,
        )

        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )

        self.dl_iter = cycle(self.dl)
        self.device = self.accelerator.device

        self.seg_model.to(self.device)

        self.optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, seg_model.parameters()),
            lr=lr
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=100 * 8,
            num_training_steps=15000 * 8,
        )

        (
            self.dl_iter,
            self.seg_model,
            self.optim,
            self.scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.seg_model,
            self.optim,
            self.scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if os.path.exists(self.results_folder):
            print('---------Warning, {} exists !!!!!!--------------------'.format(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def print(self, msg):
        """Print only on main process (accelerator-aware)."""
        self.accelerator.print(msg)

    @property
    def is_main(self):
        """Whether this process is the global main process."""
        return self.accelerator.is_main_process

    def train_step(self):
        """
        Single training step: fetch batch -> forward -> backward -> clip -> step.

        Returns:
            logs: dict containing scalar 'loss' for this step (accumulated).
        """
        start_t = time.time()
        device = self.device

        steps = int(self.steps.item())

        self.seg_model.train()

        logs = {}

        video, mask, embed_dict = next(self.dl_iter)
        device = self.device
        video = video.to(device)
        mask = mask.to(device)
        embed_dict = {k: v.to(device) for k, v in embed_dict.items()}

        with self.accelerator.autocast():
            if self.accelerator.state.deepspeed_plugin is not None:
                if self.accelerator.mixed_precision == "bf16":
                    video = video.bfloat16()
            logits, loss_dict = self.seg_model(video,
                                               mask_target=mask, embed_dict=embed_dict)
            loss = loss_dict['total']
        self.accelerator.backward(loss)
        accum_log(logs, {'loss': loss.item()})
        if exists(self.max_grad_norm):
            grad_norm = self.accelerator.clip_grad_norm_(self.seg_model.parameters(), self.max_grad_norm)
        else:
            grad_norm = self.accelerator.clip_grad_norm_(self.seg_model.parameters(), float('inf'))

        self.optim.step()
        self.optim.zero_grad()
        cur_lr = self.optim.param_groups[0]['lr']
        self.print('{}: loss: {:4f}, bce loss: {:.4f}, dice loss: {:.4f}, Grad norm: {:.4f}, lr: {:.6f}, time: {:3f}s'.format(
            steps, logs['loss'], loss_dict['bce'].item(), loss_dict['dice'].item(), grad_norm, cur_lr, time.time() - start_t))
        self.scheduler.step()

        if not (steps % self.save_model_every) and steps != 0:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.seg_model)
            state_dict = self.accelerator.get_state_dict(unwrapped_model, unwrap=False)
            if self.is_main:
                model_path = str(self.results_folder / f'VisionEncoder.{steps}.pt')
                self.accelerator.save(state_dict, model_path)
                self.print(f'{steps}: saving model to {str(self.results_folder)}')
            self.accelerator.wait_for_everyone()

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        """
        Main training loop. Runs train_step() repeatedly until num_train_steps.

        Args:
            log_fn: optional callable(logs) invoked after each step for custom logging.
        """
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
            gc.collect()
            torch.cuda.empty_cache()

        self.print('training complete')
