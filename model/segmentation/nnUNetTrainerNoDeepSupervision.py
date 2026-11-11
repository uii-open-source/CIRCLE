import torch
import torch.nn as nn
import math
from torch import autocast
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.Networks.segmodel import effunet3d_xl
from torch.nn.parallel import DistributedDataParallel as DDP


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    """
    nnU-Net trainer variant without deep supervision.
    Uses an EfficientUNet3D-XL backbone with AdamW + cosine schedule,
    and keeps the default nnU-Net training/validation pipeline.
    """

    def _build_loss(self):
        """
        Build segmentation loss according to label configuration.
        Returns:
            Combined Dice + BCE loss for region-based labels, or
            Dice + CE loss for class-based labels.
        """
        # Region-based setup (multi-label): use Dice + BCE
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        # Class-based setup (single-label): use Dice + CE
        else:
            loss = DC_and_CE_loss(
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        return loss

    def _get_deep_supervision_scales(self):
        """
        Disable deep supervision by returning None.
        """
        return None

    def configure_optimizers(self):
        """
        Configure optimizer and learning-rate scheduler.
        Returns:
            optimizer: AdamW optimizer
            lr_scheduler: LambdaLR with warmup + cosine decay
        """
        # AdamW optimizer for stable training on 3D segmentation
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        # Learning-rate schedule setup
        warmup_epochs = 10
        total_epochs = 2000

        def lr_scale(epoch: int) -> float:
            """
            Per-epoch LR scaling factor:
            - linear warmup for early epochs
            - cosine decay afterwards
            """
            # Linear warmup to avoid unstable early updates
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

        print("=" * 20)
        print("Use AdamW")
        print("=" * 20)
        return optimizer, lr_scheduler


    def initialize(self):
        """
        Initialize network, optimizer/scheduler, distributed wrapper, and loss.
        This method should only be called once per trainer instance.
        """
        if not self.was_initialized:
            # Infer input channels from nnU-Net plans + dataset metadata
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            # Determine output channels from labels/regions definition
            num_output_channels = len(self.label_manager.foreground_regions if self.label_manager.has_regions
                                      else self.label_manager.all_labels)

            # Build segmentation network
            self.network = effunet3d_xl(num_classes=num_output_channels).to(self.device)
            print("=" * 20)
            print(
                f"now use effnet with {self.num_input_channels} input channels and {num_output_channels} output channels")
            print("=" * 20)

            # Build optimizer and learning-rate scheduler
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # If running distributed training, convert BN and wrap with DDP
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            # Build task loss and finalize initialization
            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Kept for interface compatibility. No-op because deep supervision is disabled.
        Args:
            enabled: unused flag
        """
        pass

    def train_step(self, batch: dict) -> dict:
        """
        Execute one training iteration.
        Args:
            batch: dictionary containing 'data' and 'target'
        Returns:
            dict with detached scalar loss for logging
        """
        # Fetch mini-batch tensors
        data = batch['data']
        target = batch['target']

        # Move inputs/targets to current device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Clear previous gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Mixed precision on CUDA, full precision otherwise
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        # Backpropagation with/without grad scaler
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Gradient clipping to stabilize optimization
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        Execute one validation iteration and collect hard metrics.
        Args:
            batch: dictionary containing 'data' and 'target'
        Returns:
            dict with loss and per-class/region tp, fp, fn statistics
        """
        # Fetch and move batch to device
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Keep gradient buffers clean even in validation loop
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass and validation loss
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
        axes = [0] + list(range(2, output.ndim))

        # Convert logits to hard one-hot segmentation prediction
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        # Build ignore mask if ignore label exists
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        # Compute hard true/false positives/negatives
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        # Move metrics to CPU numpy for nnU-Net aggregation
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        # Exclude background channel for class-based setup
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}