#!/usr/bin/env python
"""
Training Script
"""

import os
import argparse
import random
import time
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# path setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.model import create_model
from src.dataset import create_dataloaders


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_ddp(rank, world_size):
    # DDP init
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    # nccl config
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['NCCL_HEARTBEAT_TIMEOUT_SEC'] = '300'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    torch.cuda.set_device(rank)

    # test comms
    if rank == 0:
        print("Testing NCCL communication...")
    test_tensor = torch.ones(1, device=rank)
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        expected_sum = world_size
        if test_tensor.item() == expected_sum:
            print(f"NCCL communication test passed: {test_tensor.item()} == {expected_sum}")
        else:
            print(f"NCCL communication test failed: {test_tensor.item()} != {expected_sum}")


def cleanup_ddp():
    dist.destroy_process_group()


def compute_masked_loss(pred, target, mask, loss_fn):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    if mask.dim() == 3 and pred.dim() == 4:
        mask = mask.unsqueeze(1)
    pred_masked = pred[mask]
    target_masked = target[mask]
    return loss_fn(pred_masked, target_masked)


def ensure_mask_dimensions(mask, reference_tensor):
    if mask.dim() == 3 and reference_tensor.dim() == 4:
        return mask.unsqueeze(1)
    return mask


def compute_metrics(pred, target, mask):
    mask = ensure_mask_dimensions(mask, pred)
    
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    
    pred_masked = pred[mask].cpu()
    target_masked = target[mask].cpu()
    
    # MAE
    mae = torch.abs(pred_masked - target_masked).mean().item()
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_masked - target_masked) ** 2)).item()
    
    # R2 Score
    ss_res = torch.sum((target_masked - pred_masked) ** 2)
    ss_tot = torch.sum((target_masked - torch.mean(target_masked)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    
    return mae, rmse, r2


def analyze_gradients_and_weights(model, step, log_path):
    results = []
    health_issues = []
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # weights
            weight_data = param.data.cpu()
            weight_stats = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'layer': name,
                'type': 'weight',
                'mean': weight_data.mean().item(),
                'std': weight_data.std().item(),
                'min': weight_data.min().item(),
                'max': weight_data.max().item(),
                'zero_percent': (weight_data == 0).float().mean().item() * 100,
                'norm': weight_data.norm().item()
            }
            
            writer.writerow([weight_stats[k] for k in 
                           ['timestamp', 'step', 'layer', 'type', 'mean', 'std', 'min', 'max', 'zero_percent', 'norm']])
            
            # grads
            if param.grad is not None:
                grad_data = param.grad.cpu()
                grad_stats = {
                    'timestamp': datetime.now().isoformat(),
                    'step': step,
                    'layer': name,
                    'type': 'gradient',
                    'mean': grad_data.mean().item(),
                    'std': grad_data.std().item(),
                    'min': grad_data.min().item(),
                    'max': grad_data.max().item(),
                    'zero_percent': (grad_data == 0).float().mean().item() * 100,
                    'norm': grad_data.norm().item()
                }
                
                writer.writerow([grad_stats[k] for k in 
                               ['timestamp', 'step', 'layer', 'type', 'mean', 'std', 'min', 'max', 'zero_percent', 'norm']])
                
                # health checks
                grad_norm = grad_stats['norm']
                zero_percent = grad_stats['zero_percent']
                
                if grad_norm < 1e-7:
                    health_issues.append(f"WARN {name}: Vanishing gradients (norm: {grad_norm:.2e})")
                elif grad_norm > 100:
                    health_issues.append(f"ERR {name}: Exploding gradients (norm: {grad_norm:.2f})")
                elif zero_percent > 50:
                    health_issues.append(f"WARN {name}: Many zero gradients ({zero_percent:.1f}%)")
    
    return health_issues


class CheckpointManager:
    def __init__(self, checkpoint_dir, config, rank=0):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.rank = rank
        
        self.time_interval = timedelta(hours=config['training']['checkpoint'].get('time_interval_hours', 4))
        self.last_checkpoint_time = datetime.now()
        
        self.total_steps = 0
        self.best_mae = float('inf')
        self.checkpoints_kept = []
        
        self.log_path = self.checkpoint_dir / 'training_diagnostics.csv'
        
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.log_path.exists():
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'step', 'layer', 'type', 'mean', 'std', 'min', 'max', 'zero_percent', 'norm'])
    
    def should_checkpoint(self, force=False):
        """Check if we should create a checkpoint"""
        if force:
            return True
        
        time_since_last = datetime.now() - self.last_checkpoint_time
        return time_since_last >= self.time_interval
    
    def save_checkpoint(self, model, optimizer, scheduler, scaler, metrics):
        if self.rank != 0:
            return
        
        # diagnostics
        health_issues = analyze_gradients_and_weights(model, self.total_steps, self.log_path)
        
        checkpoint = {
            'total_steps': self.total_steps,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict(),
            'metrics': metrics,
            'best_mae': self.best_mae,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'health_issues': health_issues
        }
        
        # save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.total_steps}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoints_kept.append({
            'path': checkpoint_path,
            'step': self.total_steps,
            'mae': metrics.get('val_mae', float('inf'))
        })
        
        # save best
        current_mae = metrics.get('val_mae', float('inf'))
        if current_mae < self.best_mae:
            self.best_mae = current_mae
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model (MAE: {self.best_mae:.4f}m) saved to {best_path}")
        
        self._cleanup_checkpoints()
        
        self.last_checkpoint_time = datetime.now()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        # Disabled to keep all checkpoints
        return
        if len(self.checkpoints_kept) <= 4:
            return
        
        self.checkpoints_kept.sort(key=lambda x: x['step'], reverse=True)
        
        # keep 3 recent
        to_keep = self.checkpoints_kept[:3]
        
        # keep best
        best_checkpoint = min(self.checkpoints_kept, key=lambda x: x['mae'])
        if best_checkpoint not in to_keep:
            to_keep.append(best_checkpoint)
        
        # delete others
        for checkpoint in self.checkpoints_kept:
            if checkpoint not in to_keep:
                try:
                    checkpoint['path'].unlink()
                    print(f"Cleaned up old checkpoint: {checkpoint['path'].name}")
                except FileNotFoundError:
                    pass
        
        self.checkpoints_kept = to_keep
    
    def update_step(self):
        """Update step counter"""
        self.total_steps += 1
    
    def load_checkpoint(self, checkpoint_path, model, optimizer, scaler):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.rank}')
        
        # strict load
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        self.best_mae = checkpoint.get('best_mae', float('inf'))
        self.last_checkpoint_time = datetime.now()
        
        if self.rank == 0:
            print(f"Checkpoint loaded: step {self.total_steps}, best MAE: {self.best_mae:.4f}m")
        
        return {'scheduler_state_dict': checkpoint.get('scheduler_state_dict')}

def checkpoint_validation(model, val_loader, loss_fn, rank, subset_size=30000):
    if rank != 0:
        return 0.0, 0.0, 0.0, 0.0
    
    model.eval()
    total_loss = 0
    num_batches = 0
    samples_processed = 0
    
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    total_weight = 0.0

    print(f"Checkpoint validation: Processing up to {subset_size} samples...")
    
    try:
        with torch.no_grad():
            for data in val_loader:
                image = data['image'].to(rank)
                chm = data['chm'].to(rank)
                valid_mask = data['valid_mask'].to(rank)
                continuous = data['continuous'].to(rank)
                nlcd_idx = data['nlcd_idx'].to(rank)
                ecoregion_idx = data['ecoregion_idx'].to(rank)
                
                # forward
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(image, continuous, nlcd_idx, ecoregion_idx)
                    loss = compute_masked_loss(pred, chm, valid_mask, loss_fn)
                
                batch_mae, batch_rmse, batch_r2 = compute_metrics(pred, chm, valid_mask)
                
                # accumulation
                batch_weight = valid_mask.sum().item()
                if batch_weight > 0:
                    total_mae += batch_mae * batch_weight
                    total_rmse += batch_rmse**2 * batch_weight # Accumulate squared error for RMSE
                    total_r2 += batch_r2 * batch_weight
                    total_weight += batch_weight

                total_loss += loss.item()
                num_batches += 1
                samples_processed += len(data['image'])
                
                if samples_processed >= subset_size:
                    break
    
    except Exception as e:
        print(f"Error during checkpoint validation: {e}")
        return 0.0, float('inf'), float('inf'), 0.0
    
    if total_weight == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Compute final weighted average metrics
    mae = total_mae / total_weight
    rmse = np.sqrt(total_rmse / total_weight)
    r2 = total_r2 / total_weight
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    print(f"Validation complete: {samples_processed} samples")
    print(f"   MAE: {mae:.4f}m, RMSE: {rmse:.4f}m, R²: {r2:.4f}")
    
    return avg_loss, mae, rmse, r2


class CheckpointReduceLROnPlateau:
    def __init__(self, optimizer, patience_checkpoints=3, factor=0.5, min_lr=1e-7):
        self.optimizer = optimizer
        self.patience_checkpoints = patience_checkpoints
        self.factor = factor
        self.min_lr = min_lr
        self.best_metric = float('inf')
        self.patience_counter = 0

    def step(self, metric):
        if metric < self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience_checkpoints:
            self._reduce_lr()
            self.patience_counter = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"ReduceLROnPlateau: Reduced LR from {old_lr:.2e} to {new_lr:.2e}")

    def state_dict(self):
        return {'best_metric': self.best_metric, 'patience_counter': self.patience_counter}

    def load_state_dict(self, state_dict):
        self.best_metric = state_dict['best_metric']
        self.patience_counter = state_dict['patience_counter']


def load_pretrained_unet_weights(model, checkpoint_path, rank):
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Warning: Pre-trained weights not found at {checkpoint_path}. Skipping.")
        return

    if rank == 0:
        print(f"Loading pre-trained UNet weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
    
    if 'model_state_dict' not in checkpoint:
        if rank == 0:
            print("Warning: 'model_state_dict' not found in checkpoint. Unable to load weights.")
        return
        
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.module.state_dict()
    
    # strip unet_core prefix
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('unet_core.'):
            new_key = k.replace('unet_core.', '')
            if new_key in model_dict:
                new_state_dict[new_key] = v
            else:
                 if rank == 0:
                    print(f"Warning: Key '{new_key}' not found in current model.")

    if not new_state_dict:
        if rank == 0:
            print("Warning: No matching UNet weights found in the checkpoint. Check prefixes.")
        return

    # strict=False for partial load
    model.module.load_state_dict(new_state_dict, strict=False)
    
    if rank == 0:
        print(f"Successfully loaded {len(new_state_dict)} UNet layers from pre-trained model.")


def train_cycle(model, train_loader, val_loader, optimizer, scaler, loss_fn, scheduler, checkpoint_manager, writer, rank, world_size, config, start_step=0):
    model.train()
    max_steps = config['training'].get('max_steps', 1000000)
    warmup_steps = config['training'].get('warmup_steps', 10000)
    target_lr = float(config['training']['learning_rate'])
    step = start_step
    train_iter = iter(train_loader)

    if rank == 0:
        print(f"Starting training from step {start_step} up to {max_steps} steps.")

    while step < max_steps:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)
        
        # warmup
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            current_lr = target_lr * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        image = data['image'].to(rank)
        chm = data['chm'].to(rank)
        valid_mask = data['valid_mask'].to(rank)
        continuous = data['continuous'].to(rank)
        nlcd_idx = data['nlcd_idx'].to(rank)
        ecoregion_idx = data['ecoregion_idx'].to(rank)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(image, continuous, nlcd_idx, ecoregion_idx)
            loss = compute_masked_loss(pred, chm, valid_mask, loss_fn)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        checkpoint_manager.update_step()
        step += 1

        if rank == 0 and step % 100 == 0:
            print(f"Step {step}/{max_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
            if writer:
                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('LR', current_lr, step)

        if checkpoint_manager.should_checkpoint():
            val_loss, val_mae, val_rmse, val_r2 = checkpoint_validation(model, val_loader, loss_fn, rank, config['training']['checkpoint']['subset_val_size'])
            if rank == 0:
                metrics = {
                    'step': step,
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'lr': current_lr
                }
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, scaler, metrics)
                if writer:
                    writer.add_scalar('Loss/val', val_loss, step)
                    writer.add_scalar('MAE/val', val_mae, step)
                    writer.add_scalar('RMSE/val', val_rmse, step)
                    writer.add_scalar('R2/val', val_r2, step)
                    writer.flush()
                if step >= warmup_steps:
                    scheduler.step(val_mae)
            model.train()


def train_worker(rank, world_size, config, args):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    if rank == 0:
        print(f"Thread limiting enabled.")

    setup_ddp(rank, world_size)
    set_seed(config['training']['seed'] + rank)
    
    model = create_model(config).to(rank)
    
    if config['training'].get('freeze_unet', False):
        if rank == 0: print("Freezing UNet weights.")
        model.freeze_unet_weights()
    
    if config['training'].get('freeze_film', False):
        if rank == 0: print("Freezing FiLM weights.")
        model.freeze_film_weights()
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=config['distributed'].get('find_unused_parameters', False))

    if args.pretrained_weights:
        if rank == 0:
            print(f"Loading pretrained weights from: {args.pretrained_weights}")
        # load checkpoint
        checkpoint = torch.load(args.pretrained_weights, map_location=f'cuda:{rank}')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # remove module prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            if rank == 0:
                print("Removing 'module.' prefix from pretrained weights")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        # strict=False
        model.module.load_state_dict(state_dict, strict=False)
        
        if rank == 0:
            print(f"Loaded pretrained weights (strict=False)")
            
    elif 'load_pretrained_unet_weights' in config['training']:
        load_pretrained_unet_weights(model, config['training']['load_pretrained_unet_weights'], rank)

    train_loader, val_loader, _, _ = create_dataloaders(config, rank, world_size)
    
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    
    loss_fn = nn.HuberLoss(delta=config['training']['huber_delta'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['amp']['enabled'])
    
    writer = None
    if rank == 0:
        log_dir = Path(config['logging']['log_dir']) / args.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        with open(log_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    checkpoint_manager = CheckpointManager(checkpoint_dir=Path(config['training']['checkpoint']['save_dir']) / args.experiment_name, config=config, rank=rank)
    
    scheduler = CheckpointReduceLROnPlateau(optimizer, patience_checkpoints=config['training']['scheduler']['plateau']['patience_checkpoints'], factor=config['training']['scheduler']['plateau']['factor'], min_lr=config['training']['scheduler']['plateau']['min_lr'])

    start_step = 0
    if args.resume_from:
        if rank == 0: print(f"Resuming from checkpoint: {args.resume_from}")
        resume_state = checkpoint_manager.load_checkpoint(args.resume_from, model, optimizer, scaler)
        if resume_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        start_step = checkpoint_manager.total_steps

    train_cycle(model, train_loader, val_loader, optimizer, scaler, loss_fn, scheduler, checkpoint_manager, writer, rank, world_size, config, start_step=start_step)
    
    if rank == 0 and writer:
        writer.close()
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train Canopy Height Model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment for logging')
    parser.add_argument('--resume_from', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_weights', type=str, default='', help='Path to pretrained model weights to load (starts new session)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1 and rank == 0:
        print(f"Starting DDP training with {world_size} GPUs")
    elif world_size == 1:
        print("Starting single-GPU training")

    train_worker(rank, world_size, config, args)

    if rank == 0:
        print("Training completed!")


if __name__ == '__main__':
    main()
