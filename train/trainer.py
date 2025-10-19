import gc
import numpy as np
import os
import time
import torch

# Accelerate provides convenient training tools across multiple GPUs/distributed/mixed precision, etc
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch import nn
from torch.utils.data import DataLoader

# Used to export the complete state dictionary from FullySarddDataParallel (FSDP) (can be offloaded to CPU when saving GPU memory)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, StateDictType

# Customize datasets and optimizer to obtain functions
from train.dataset import CIRCLEDataset
from train.utils import get_optimizer


def exists(val):
    """
    Small utility: return True if val is not None.
    Used to conditionally apply gradient clipping when max_grad_norm is provided.
    """
    return val is not None

def noop(*args, **kwargs):
    """
    No-op function used as default log_fn in train() so user can optionally pass a logging callback.
    """
    pass

def cycle(dl):
    """
    Make an infinite iterator from a DataLoader.
    This yields batches forever in a loop, useful when number of steps >> number of epochs.
    """
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    """
    Helper to get a yes/no answer from stdin. Not used in current code but kept for interactive prompts.
    """
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    """
    Accumulate numeric logging metrics into a dictionary.
    Adds values from new_logs to existing entries in log (or creates them).
    """
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# Trainer inspired by CT-CLIP (https://github.com/ibrahimethemhamamci/CT-CLIP)
class CIRCLETrainer(nn.Module):
    """
    High-level trainer class that wraps:
      - dataset and dataloader creation
      - accelerator (mixed-precision, distributed) setup
      - optimizer creation
      - training step (forward, backward, step, logging)
      - periodic checkpoint saving (supports FSDP full-state export)
    The class is implemented as a nn.Module for convenience but it primarily orchestrates training.
    """
    def __init__(
        self,
        circle_model,
        tokenizer,
        data_folder,
        label_csv,
        lung_center_csv,
        report_csv,
        num_train_steps,
        batch_size,
        lr = 1.25e-5,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = '/results',
        num_workers = 8,
        accelerate_kwargs: dict = dict()
    ):
        """
        Constructor arguments:
          - circle_model: instance of the CIRCLE model (image+text)
          - tokenizer: HuggingFace tokenizer for text batching
          - data_folder, label_csv, lung_center_csv, report_csv: dataset inputs
          - num_train_steps: total training steps to run
          - batch_size: batch size passed to DataLoader
          - lr, wd: optimizer hyperparameters
          - max_grad_norm: gradient clipping threshold
          - save_results_every / save_model_every: periodic save frequencies (in steps)
          - results_folder: directory for saving checkpoints / outputs
          - num_workers: DataLoader workers
          - accelerate_kwargs: extra kwargs forwarded to Accelerator
        """
        super().__init__()

        # configure DDP kwargs (find_unused_parameters may be required when some parameters are not touched every step)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # configure process group init timeout (long timeout to accommodate heavy I/O or slow nodes)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))

        # create Accelerator instance with mixed precision FP16 and our kwargs handlers
        # Accelerator handles device placement, distributed setup, and preparing models/optimizers/dataloaders
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], mixed_precision='fp16', **accelerate_kwargs)

        # model and tokenizer references
        self.circle = circle_model
        self.tokenizer=tokenizer
       
        # track training steps in a registered buffer so the state can be moved with the module if needed
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        # collect all model parameters into a set for optimizer creation
        all_parameters = set(circle_model.parameters())

        # build optimizer using the helper (get_optimizer likely returns AdamW or similar)
        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        # gradient clipping value
        self.max_grad_norm = max_grad_norm
        self.lr=lr

        # initialize dataset (loads CSVs and prepares sample list)
        self.ds = CIRCLEDataset(data_folder, label_csv, lung_center_csv, report_csv)

        # build DataLoader: shuffling enabled (typical for training)
        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
        )

        # prepare infinite iterator over the dataloader (so train_step can call next() directly)
        self.dl_iter=cycle(self.dl)

        # get device from accelerator (handles CPU / single-GPU / multi-GPU placement)
        self.device = self.accelerator.device

        # move model to device (Accelerator will handle further distribution/wrapping)
        self.circle.to(self.device)

        # call accelerator.prepare to properly wrap/prepare the iterator, model, and optimizer
        # after this, objects are replaced by device/distributed-safe versions
        (
            self.dl_iter,
            self.circle,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.circle,
            self.optim,
        )

        # saving configuration
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def save(self, path):
        """
        Save a checkpoint containing:
          - model state dict (accelerator.get_state_dict handles wrapped models)
          - optimizer state dict
        Only run on the local main process to avoid race conditions.
        """
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.circle),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        """
        Load checkpoint from path and restore model + optimizer states.
        Note: expects 'path' to be a pathlib.Path-like object with .exists(). Caller must ensure correct type.
        """
        assert path.exists()
        pkg = torch.load(path)

        # unwrap accelerator-wrapped model to get underlying module and load state_dict
        circle = self.accelerator.unwrap_model(self.circle)
        circle.load_state_dict(pkg['model'])

        # restore optimizer state as well
        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        """
        Wrapper around accelerator.print so prints happen only on main process depending on accelerator settings.
        """
        self.accelerator.print(msg)


    @property
    def is_main(self):
        """
        Convenience property: whether this process is the main process (across multi-process setups).
        """
        return self.accelerator.is_main_process

    def train_step(self):
        """
        Execute a single training step:
          - fetch batch (image, text, label)
          - tokenize text with tokenizer -> move to device
          - forward pass through CIRCLE model (under autocast for mixed precision)
          - backward (accelerator.backward)
          - optional gradient clipping
          - optimizer step and zero_grad
          - logging and periodic checkpoint saving (supports FSDP FULL_STATE_DICT export)
        Returns logs dictionary containing numeric metrics for this step.
        """
        start_t = time.time()
        device = self.device

        # read current step count (stored as tensor buffer)
        steps = int(self.steps.item())

        self.circle.train()

        # logs aggregator for this step
        logs = {}

        # fetch next batch from infinite iterator (dl_iter was prepared by accelerator)
        image, text, label = next(self.dl_iter)

        # move tensors to device
        device=self.device
        image = image.to(device)
        label = label.float().to(device)

        # text is a Python list of raw strings from dataset; we convert via tokenizer
        text = list(text)
        # tokenizer returns a dict with input_ids, attention_mask, etc. We ask it to return PyTorch tensors,
        # pad/truncate to max_length=512 and then move the tensors to device
        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        
        # Mixed precision forward: accelerator.autocast ensures operations run in fp16 where safe
        with self.accelerator.autocast():
            # circle(...) returns loss, cls_loss, clip_loss (see model.forward)
            loss, cls_loss, clip_loss = self.circle(text_tokens, image, label, device)
        
        # Backward pass using accelerator's backward to handle mixed precision scaling
        self.accelerator.backward(loss)

        # record scalar loss in logs
        accum_log(logs, {'loss': loss.item()})

        # optional gradient clipping (only if max_grad_norm is not None)
        if exists(self.max_grad_norm):
            # accelerator.clip_grad_norm_ wraps different backends appropriately
            self.accelerator.clip_grad_norm_(self.circle.parameters(), self.max_grad_norm)

        # optimizer step updates parameters (optimizer is prepared by accelerator)
        self.optim.step()
        # zero gradients in optimizer
        self.optim.zero_grad()

        # print per-step log: step index, losses, and elapsed time
        # note: cls_loss and clip_loss are tensors, so .item() is used for numeric values
        self.print('{}: loss: {:4f}, cls loss: {:.4f}, clip loss: {:.4f} time: {:3f}s'.format(
            steps, logs['loss'], cls_loss.item(), clip_loss.item(), time.time() - start_t))

        # Periodic model checkpointing:
        # Every save_model_every steps (except step 0) export a full state dict.
        # If using FSDP, export FULL_STATE_DICT with possible offload to CPU to avoid OOM.
        if not (steps % self.save_model_every) and steps != 0:
            # Prepare FSDP full state dict config (offload full state to CPU, rank0_only True)
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # Use context manager to temporarily change state_dict_type to FULL_STATE_DICT for FSDP
            with torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(self.circle,
                                                                                 StateDictType.FULL_STATE_DICT,
                                                                                 full_state_dict_config):
                # get state dict (accelerator.get_state_dict will return the model state appropriately)
                state_dict = self.accelerator.get_state_dict(self.circle, unwrap=False)
            if self.is_main:
                # construct model path (results_folder is a string; using f-string with pathlib-like semantics)
                model_path = str(self.results_folder / f'circle.{steps}.pt')
                # accelerator.save handles saving in distributed environment
                self.accelerator.save(state_dict, model_path)
                self.print(f'{steps}: saving model to {str(self.results_folder)}')

        # increment step counter buffer
        self.steps += 1
        return logs



    def train(self, log_fn=noop):
        """
        Run the main training loop until steps >= num_train_steps.
        At each iteration, call train_step and optionally a user-supplied log function.
        Will also call garbage collection and empty CUDA cache between steps to reduce memory fragmentation.
        """
        while self.steps < self.num_train_steps:
            t = time.time()
            logs = self.train_step()
            # user-provided logging function can handle logs (e.g., send to TensorBoard)
            log_fn(logs)
            # explicit garbage collection to reduce memory spikes in long-running loops
            gc.collect()
            torch.cuda.empty_cache()

        self.print('training complete')
