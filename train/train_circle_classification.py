import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from easydict import EasyDict as edict
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import CIRCLEClassificationDataset
from model.efficient_net import effnetv2_xl_cls


def get_auc(all_labels, all_probs, n_classes=None):
    """
    Compute macro-averaged AUC and per-class AUC scores.
    Args:
        all_labels: list of int ground-truth class indices.
        all_probs: list of numpy arrays of shape (batch, n_classes) with predicted probabilities.
        n_classes: int, number of classes; inferred from all_labels if not provided.
    Returns:
        auc_score: float macro-averaged AUC, or None if computation fails.
        per_class_auc: dict mapping class index to its AUC value (NaN if class absent).
    """
    if n_classes is None:
        n_classes = np.max(np.array(all_labels)) + 1
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)  # shape: [N, n_classes]

    auc_score = None
    per_class_auc = {}  # {class_id: auc}
    try:
        if len(np.unique(all_labels)) < 2:
            print("Warning: Only one class present in y_true. AUC is undefined.")
        else:
            if n_classes == 2:
                # Binary classification: use probability of positive class
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
                per_class_auc[1] = auc_score  # same as above
            else:
                # Multi-class: compute per-class OvR AUC then average
                y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
                for i in range(n_classes):
                    if y_bin[:, i].sum() == 0:
                        per_class_auc[i] = float('nan')  # 或设为 None
                        print(f"Warning: Class {i} not present in y_true. AUC set to NaN.")
                    else:
                        per_class_auc[i] = roc_auc_score(y_bin[:, i], all_probs[:, i])
                valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
                auc_score = np.mean(valid_aucs) if valid_aucs else None
                # Cross-validate with sklearn's built-in macro OvR AUC
                auc_score2 = roc_auc_score(
                    all_labels, all_probs,
                    multi_class="ovr",
                    average="macro"
                )
                if abs(auc_score - auc_score2) > 0.01:
                    print('--------Error:', auc_score, auc_score2)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc_score = None

    return auc_score, per_class_auc


class IOStream():
    """
    Simple logger that writes to both stdout and a text file simultaneously.
    """
    def __init__(self, path):
        """
        Open log file for appending.
        Args:
            path: path to the log file.
        """
        self.f = open(path, 'a')

    def cprint(self, text):
        """
        Print text to stdout and append it to the log file.
        Args:
            text: string to log.
        """
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        """Close the underlying log file."""
        self.f.close()
        
        
def load_model_ckpt(model, ckpt_path):
    """
    Load pretrained weights into a model, handling DataParallel "module." prefix.
    Args:
        model: PyTorch model to load weights into.
        ckpt_path: path to the checkpoint file.
    Returns:
        model with loaded weights
    """
    # Load checkpoint to CPU
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    # Remove 'module.' prefix if present (models trained with DataParallel)
    for key, val in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val
    model.load_state_dict(new_state_dict, strict=True)
    return model


def create_model(num_class, pretrain_path=None):
    """
    Build EfficientNetV2-XL classification model and optionally load pretrained weights.
    Args:
        num_class: int, number of output classes.
        pretrain_path: optional path to a CIRCLE checkpoint; visual_transformer weights are extracted.
    Returns:
        model: initialized EfficientNetV2-XL classification model
    """
    model = effnetv2_xl_cls(
        num_classes=num_class
    )
    if pretrain_path is not None:
        # Extract visual_transformer weights from a full CIRCLE checkpoint
        state_dict = torch.load(pretrain_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'visual_transformer' in key:
                new_state_dict[key.replace('visual_transformer.', '')] = value
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
    return model


class Trainer:
    """
    Training manager for the CIRCLE classification model.
    Handles logging, model/optimizer/scheduler setup, training loop, and validation.
    """
    def __init__(self, cfg):
        """
        Initialize trainer with configuration.
        Args:
            cfg: EasyDict containing all training hyperparameters and paths.
        """
        self.cfg = cfg
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.device_ids
        # Mixed-precision gradient scaler
        self.scaler = GradScaler()
        # Epoch and step counters
        self.curr_epoch = 0
        self.curr_step = 0
        
    def set_log(self):
        """
        Create timestamped log directory with checkpoint and tensorboard subdirectories,
        and initialize the IOStream logger.
        """
        now = datetime.now()
        curr_time = now.strftime(r'%m%d%H%M')
        
        # Build timestamped log directory name
        self.log_dir = os.path.join(
            self.cfg.log_dir, 
            "classification_{}_{}_{}_{}".format(
                curr_time, 
                self.cfg.arch_name,
                self.cfg.task_type,
                self.cfg.optm_type,
        ))
        os.makedirs(self.log_dir, exist_ok=True)
       
        # Create checkpoint subdirectory
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        
        # Create TensorBoard subdirectory
        tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        
        # Initialize text logger
        log_path = os.path.join(self.log_dir, "training_log.txt")
        self.logger = IOStream(log_path)
        
    def set_model(self):
        """
        Build the classification model, wrap with DataParallel if multiple GPUs are specified,
        and move to CUDA.
        """
        model = create_model(self.cfg.num_class)
        device_id_list = [x.strip() for x in str(self.cfg.device_ids).split(',') if x.strip()]
        # Wrap with DataParallel when more than one GPU is available
        if len(device_id_list) > 1:
            model = nn.DataParallel(model)
        model.cuda()
        self.model = model

    def set_loss_fn(self):
        """Initialize cross-entropy loss function on CUDA."""
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    def set_optimizer(self):
        """
        Initialize SGD or AdamW optimizer according to cfg.optm_type.
        """
        if self.cfg.optm_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.cfg.lr, 
                momentum=0.9, 
                weight_decay=1e-4
            )
        elif self.cfg.optm_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.cfg.lr, 
                weight_decay=self.cfg.weight_decay,
            )
        self.optimizer = optimizer

    def set_scheduler(self):
        """
        Initialize cosine annealing learning rate scheduler.
        """
        scheduler = CosineAnnealingLR(
            self.optimizer, 
            self.cfg.total_epoch, 
            eta_min=self.cfg.lr/100,
        )
        self.scheduler = scheduler
        
    def save_model(self):
        """
        Save current model state dict to the checkpoint directory.
        Filename encodes architecture name, current epoch, and step.
        """
        print('='*40)
        model_save_path = os.path.join(
            self.checkpoint_dir, 
            "{}_e{:d}_step{:d}.pth".format(
                self.cfg.arch_name,
                self.curr_epoch+1, 
                self.curr_step,
            )
        )
        torch.save(
            self.model.state_dict(), 
            model_save_path,
        )
        print("Model saved to:", model_save_path)
        
    def write_train_log(self, losses, loss_range):
        """
        Log averaged training loss over the last loss_range steps.
        Args:
            losses: list of per-step loss values accumulated so far.
            loss_range: number of recent steps to average for the log entry.
        """
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.logger.cprint(
            'Train: Epoch = {} / {}, Step = {}, lr = {:.6f}, Loss = {:.6f}'.format(
                self.curr_epoch + 1, 
                self.cfg.total_epoch, 
                self.curr_step,
                lr, 
                np.mean(losses[-loss_range:]),
        ))

    def train_one_epoch(self, train_loader):
        """
        Run one full training epoch with mixed-precision forward/backward passes.
        Args:
            train_loader: DataLoader yielding (image_tensor, label) batches.
        """
        self.model.train()
        losses = []
        with tqdm(total = len(train_loader)) as _tqdm:
            _tqdm.set_description(
                'train epoch: {}/{}'.format(
                    self.curr_epoch+1, 
                    self.cfg.total_epoch
            ))
            
            for img_tensor, labels in train_loader:
                # Periodically log and save checkpoint
                if self.curr_step % self.cfg.log_interval == 0 and self.curr_step > 0:
                    self.write_train_log(losses, self.cfg.log_interval)
                    self.save_model()
                
                self.optimizer.zero_grad()
                
                img_tensor = img_tensor.cuda()
                gt_labels = labels.cuda()
                # Forward pass with automatic mixed precision
                with autocast():  # dtype=torch.bfloat16
                    pred_logits = self.model(img_tensor)
                    loss = self.loss_fn(pred_logits, gt_labels)
                # Scaled backward pass and optimizer step
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses.append(loss.item())
                _tqdm.set_postfix(loss='{:.6f}'.format(np.mean(losses[-100:])))
                _tqdm.update(1)
                self.curr_step += 1
            
            self.write_train_log(losses, self.cfg.log_interval)

    def val_one_epoch(self, val_loader, save_model=True, description='Validation'):
        """
        Run one full validation epoch, compute AUC, and log results.
        Args:
            val_loader: DataLoader yielding (image_tensor, label) batches.
            save_model: whether to save a checkpoint before validation.
            description: string prefix used in log output.
        """
        if save_model:
            self.save_model()
        self.model.eval()
        losses = []
        with tqdm(total = len(val_loader)) as _tqdm:
            _tqdm.set_description(
                '{}: epoch={}, step={}'.format(
                    description,
                    self.curr_epoch + 1, 
                    self.curr_step,
                )
            )
            all_labels = []
            all_probs = []
            for img_tensor, label in val_loader:
                img_tensor = img_tensor.cuda()
                gt_label = label.cuda()
                
                with torch.no_grad():
                    logit = self.model(img_tensor)
                    loss = self.loss_fn(logit, gt_label)
                    losses.append(loss.item())

                # Convert logits to class probabilities
                probs = torch.softmax(logit, dim=1)  # [batch_size, n_classes]
                all_labels.extend(label.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                _tqdm.set_postfix(loss='{:.6f}'.format(loss.item()))
                _tqdm.update(1)
        
        # Compute and log macro AUC
        auc, _ = get_auc(all_labels, all_probs)
        auc_str = '{:.4f}'.format(auc) if auc is not None else 'nan'
        self.logger.cprint(
            '{}: Epoch={} / {}, Step={}, lr: {:.6f}, Loss: {:.6f}, Auc: {}'.format(
                description,
                self.curr_epoch + 1, 
                self.cfg.total_epoch, 
                self.curr_step,
                self.optimizer.state_dict()['param_groups'][0]['lr'], 
                np.mean(losses),
                auc_str
        ))

    def train(self):
        """
        Full training pipeline: build datasets/loaders, initialize components,
        then iterate over epochs running training and validation.
        """
        # Build train and validation datasets
        train_dataset = CIRCLEClassificationDataset()
        val_dataset = CIRCLEClassificationDataset()
        train_dataset.read_data_csv(self.cfg.train_data_csv)
        val_dataset.read_data_csv(self.cfg.valid_data_csv)
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=4,
        )
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=4,
        )

        print("Training on:", self.cfg.device_ids)
        # Set up logging, model, loss, optimizer, and scheduler
        self.set_log()
        self.set_model()
        self.set_loss_fn()
        self.set_optimizer()
        self.set_scheduler()
        
        # Main training loop
        start_epoch = 0
        for epoch in range(start_epoch, self.cfg.total_epoch):
            self.curr_epoch = epoch
            self.train_one_epoch(train_loader)
            self.val_one_epoch(val_loader, save_model=True, description='Validation')
            self.scheduler.step()
            
            
def train(
    task_type,
    num_class,
    train_data_csv,
    val_data_csv,
    save_dir,
    device_ids="0",
    lr=1e-4,
    optm_type="adamw",
    batch_size=2,
    weight_decay=1e-3,
    total_epoch=30,
    log_interval=2000,
):
    """
    Entry point for classification training: build config, create Trainer, and run.
    Args:
        task_type: str, task name used in log directory naming.
        num_class: int, number of output classes.
        train_data_csv: str, path to training CSV.
        val_data_csv: str, path to validation CSV.
        save_dir: str, root directory for logs and checkpoints.
        device_ids: str, comma-separated CUDA device ids, e.g. "0" or "0,1".
        lr: float, initial learning rate.
        optm_type: str, optimizer type ('sgd' or 'adamw').
        batch_size: int, training batch size.
        weight_decay: float, optimizer weight decay.
        total_epoch: int, total number of training epochs.
        log_interval: int, number of steps between log/checkpoint saves.
    """
    # Assemble training configuration
    train_cfg = edict()
    train_cfg.arch_name = 'circle'
    train_cfg.device_ids = device_ids
    train_cfg.train_data_csv = train_data_csv
    train_cfg.valid_data_csv = val_data_csv
    train_cfg.log_dir = save_dir
    train_cfg.task_type = task_type
    train_cfg.optm_type = optm_type
    train_cfg.batch_size = batch_size
    train_cfg.lr = lr
    train_cfg.weight_decay = weight_decay
    train_cfg.total_epoch = total_epoch
    train_cfg.log_interval = log_interval
    train_cfg.num_class = num_class
    
    trainer = Trainer(train_cfg)
    
    # Run training and report elapsed time
    tic = time.time()
    trainer.train()
    toc = time.time()
    print('Training completed, {:.2f} minutes'.format((toc - tic) / 60))


def parse_args():
    """Parse command-line arguments for classification training."""
    parser = argparse.ArgumentParser(description='Train CIRCLE classification model')
    parser.add_argument('--task_type', type=str, required=True, help='Task type name for logging')
    parser.add_argument('--num_class', type=int, required=True, help='Number of classes')
    parser.add_argument('--train_data_csv', type=str, required=True, help='Training csv path')
    parser.add_argument('--val_data_csv', type=str, required=True, help='Validation csv path')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--device_ids', type=str, default='0', help='CUDA visible device ids, e.g. "0" or "0,1"')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optm_type', type=str, default='adamw', choices=['sgd', 'adamw'], help='Optimizer type')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--total_epoch', type=int, default=30, help='Total training epochs')
    parser.add_argument('--log_interval', type=int, default=2000, help='Train log interval in steps')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        task_type=args.task_type,
        num_class=args.num_class,
        train_data_csv=args.train_data_csv,
        val_data_csv=args.val_data_csv,
        save_dir=args.save_dir,
        device_ids=args.device_ids,
        lr=args.lr,
        optm_type=args.optm_type,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        total_epoch=args.total_epoch,
        log_interval=args.log_interval,
    )
