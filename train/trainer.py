import gc
import numpy as np
import os
import time
import torch

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch import nn
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, StateDictType

from train.dataset import CIRCLEDataset
from train.utils import get_optimizer


def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# Trainer inspired by CT-CLIP (https://github.com/ibrahimethemhamamci/CT-CLIP)
class CIRCLETrainer(nn.Module):
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
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], mixed_precision='fp16', **accelerate_kwargs)
        self.circle = circle_model
        self.tokenizer=tokenizer
       
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(circle_model.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        self.ds = CIRCLEDataset(data_folder, label_csv, lung_center_csv, report_csv)

        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.circle.to(self.device)
        (
 			self.dl_iter,
            self.circle,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.circle,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.circle),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        assert path.exists()
        pkg = torch.load(path)

        circle = self.accelerator.unwrap_model(self.circle)
        circle.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        start_t = time.time()
        device = self.device

        steps = int(self.steps.item())

        self.circle.train()

        # logs
        logs = {}

        image, text, label = next(self.dl_iter)
        device=self.device
        image = image.to(device)
        label = label.float().to(device)
        text = list(text)
        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        
        with self.accelerator.autocast():
            loss, cls_loss, clip_loss = self.circle(text_tokens, image, label, device)
        
        self.accelerator.backward(loss)
        accum_log(logs, {'loss': loss.item()})
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.circle.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        self.print('{}: loss: {:4f}, cls loss: {:.4f}, clip loss: {:.4f} time: {:3f}s'.format(
            steps, logs['loss'], cls_loss.item(), clip_loss.item(), time.time() - start_t))

        # save model every so often
        if not (steps % self.save_model_every) and steps != 0:
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(self.circle,
                                                                                 StateDictType.FULL_STATE_DICT,
                                                                                 full_state_dict_config):
                state_dict = self.accelerator.get_state_dict(self.circle, unwrap=False)
            if self.is_main:
                model_path = str(self.results_folder / f'circle.{steps}.pt')
                self.accelerator.save(state_dict, model_path)
                self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs



    def train(self, log_fn=noop):
        while self.steps < self.num_train_steps:
            t = time.time()
            logs = self.train_step()
            log_fn(logs)
            gc.collect()
            torch.cuda.empty_cache()

        self.print('training complete')
