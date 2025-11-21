"""
Model Evaluation Script
"""

import os
import sys
import argparse
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.spatial.distance import jensenshannon

# setup paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.model import create_model
from src.dataset import CanopyHeightDataset, NLCD_CLASSES, NLCD_TO_IDX
from utils.coordinate_transform import CoordinateTransformer


def setup_ddp(rank, world_size):
    # init ddp
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


def calculate_jsd(pred, target, valid_mask, num_bins=60, height_range=(0, 60)):
    # JSD metric calculation
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return None
    
    # histograms
    pred_hist = torch.histc(pred_valid, bins=num_bins, min=height_range[0], max=height_range[1])
    target_hist = torch.histc(target_valid, bins=num_bins, min=height_range[0], max=height_range[1])
    
    pred_hist_np = pred_hist.cpu().numpy()
    target_hist_np = target_hist.cpu().numpy()
    
    # numerical stability
    epsilon = 1e-10
    pred_hist_np = pred_hist_np + epsilon
    target_hist_np = target_hist_np + epsilon
    
    # normalize
    pred_prob = pred_hist_np / pred_hist_np.sum()
    target_prob = target_hist_np / target_hist_np.sum()
    
    # get divergence
    jsd = jensenshannon(target_prob, pred_prob)
    jsd_divergence = jsd ** 2
    
    return jsd_divergence


def create_metric_accumulator_dict():
    """Creates a defaultdict of MetricAccumulators, required for pickling."""
    return defaultdict(MetricAccumulator)


class ModelEvaluator:
    def __init__(self, model, device, target_size=432, base_path=None):
        self.model = model
        self.device = device
        self.target_size = target_size
        self.base_path = base_path
        
        # nlcd mapping
        self.nlcd_names = {
            11: 'Water',
            21: 'Developed', 
            31: 'Barren',
            41: 'Forest',
            52: 'Shrubland',
            71: 'Grassland',
            81: 'Pasture/Hay',
            82: 'Cultivated',
            90: 'Wetlands'
        }
        
        self.reset_metrics()
        self.tile_stats_06m = []
        self.tile_stats_10m = []
    
    def reset_metrics(self):
        self.overall_06m = MetricAccumulator()
        self.overall_10m = MetricAccumulator()
        
        self.nlcd_06m = {cls: MetricAccumulator() for cls in NLCD_CLASSES}
        self.nlcd_10m = {cls: MetricAccumulator() for cls in NLCD_CLASSES}
        
        self.ecoregion_nlcd_06m = defaultdict(create_metric_accumulator_dict)
        self.ecoregion_nlcd_10m = defaultdict(create_metric_accumulator_dict)
    
    def resample_to_1m(self, pred_06m, original_bounds, original_shape):
        # resample 0.6m -> 1.0m
        pred_np = pred_06m.cpu().numpy().squeeze()
        
        # src transform (0.6m)
        src_transform = rasterio.transform.from_bounds(
            original_bounds[0], original_bounds[1],
            original_bounds[2], original_bounds[3],
            self.target_size, self.target_size
        )
        
        # dst transform (1.0m)
        dst_transform = rasterio.transform.from_bounds(
            original_bounds[0], original_bounds[1], 
            original_bounds[2], original_bounds[3],
            original_shape[1], original_shape[0]
        )
        
        pred_10m = np.empty(original_shape, dtype=np.float32)
        reproject(
            source=pred_np,
            destination=pred_10m,
            src_transform=src_transform,
            dst_transform=dst_transform,
            resampling=Resampling.bilinear
        )
        
        return torch.from_numpy(pred_10m).to(self.device)
    
    def calculate_metrics(self, pred, target, valid_mask):
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        if len(pred_valid) == 0:
            return None
        
        errors = pred_valid - target_valid
        mae = torch.mean(torch.abs(errors)).item()
        rmse = torch.sqrt(torch.mean(errors ** 2)).item()
        bias = torch.mean(errors).item()
        
        # R2 stats
        ss_res = torch.sum(errors ** 2).item()
        target_sum = torch.sum(target_valid).item()
        target_sq_sum = torch.sum(target_valid ** 2).item()
        
        n_pixels_tile = len(pred_valid)
        if n_pixels_tile > 1:
            ss_tot = target_sq_sum - (target_sum ** 2) / n_pixels_tile
        else:
            ss_tot = 0.0

        total_pred = torch.sum(pred_valid).item()
        total_target = torch.sum(target_valid).item()
        
        avg_pred_height = torch.mean(pred_valid).item()
        avg_obs_height = torch.mean(target_valid).item()
        
        # p95
        pred_height_p95 = torch.quantile(pred_valid, 0.95).item() if len(pred_valid) > 0 else 0
        obs_height_p95 = torch.quantile(target_valid, 0.95).item() if len(target_valid) > 0 else 0
        
        jsd = calculate_jsd(pred, target, valid_mask)
        
        return {
            'mae': mae, 'rmse': rmse, 'bias': bias,
            'ss_res': ss_res, 'ss_tot': ss_tot,
            'target_sum': target_sum, 'target_sq_sum': target_sq_sum,
            'total_pred': total_pred, 'total_target': total_target,
            'avg_pred_height': avg_pred_height, 'avg_obs_height': avg_obs_height,
            'pred_height_p95': pred_height_p95, 'obs_height_p95': obs_height_p95,
            'n_pixels': len(pred_valid),
            'jsd': jsd if jsd is not None else np.nan
        }
    
    def update_metrics(self, metrics_06m, metrics_10m, nlcd_class, ecoregion):
        if metrics_06m:
            self.overall_06m.update(metrics_06m)
        if metrics_10m:
            self.overall_10m.update(metrics_10m)
        
        if metrics_06m:
            self.nlcd_06m[nlcd_class].update(metrics_06m)
        if metrics_10m:
            self.nlcd_10m[nlcd_class].update(metrics_10m)
        
        if metrics_06m:
            self.ecoregion_nlcd_06m[ecoregion][nlcd_class].update(metrics_06m)
        if metrics_10m:
            self.ecoregion_nlcd_10m[ecoregion][nlcd_class].update(metrics_10m)
    
    def evaluate_batch(self, batch, enable_10m_evaluation=False):
        image = batch['image'].to(self.device)
        chm_06m = batch['chm'].to(self.device) 
        valid_mask_06m = batch['valid_mask'].to(self.device)
        continuous = batch['continuous'].to(self.device)
        nlcd_idx = batch['nlcd_idx'].to(self.device)
        ecoregion_idx = batch['ecoregion_idx'].to(self.device)
        chm_paths = batch['chm_path']
        
        batch_size = image.size(0)
        nlcd_classes = [NLCD_CLASSES[idx.item()] for idx in nlcd_idx]
        
        ecoregions = [ecoregion_idx[i].item() for i in range(batch_size)]
        
        # Quality checks (min 25% valid pixels)
        MIN_VALID_PIXELS = 45582
        
        samples_to_process = []
        for i in range(batch_size):
            valid_pixel_count = torch.sum(valid_mask_06m[i]).item()
            if valid_pixel_count >= MIN_VALID_PIXELS:
                samples_to_process.append(i)
        
        # Filter outliers / bad data
        filtered_samples = []
        for i in samples_to_process:
            chm_valid = chm_06m[i].squeeze(0)[valid_mask_06m[i]]
            
            if len(chm_valid) == 0:
                continue
            
            avg_obs_height = torch.mean(chm_valid).item()
            obs_height_p95 = torch.quantile(chm_valid, 0.95).item()
            height_range = obs_height_p95 - avg_obs_height
            
            # filter unrealistic p95
            if obs_height_p95 >= 97.131:
                continue
            
            # filter high avg
            if avg_obs_height > 62.0:
                continue
            
            # filter large range
            if height_range >= 37:
                continue
            
            filtered_samples.append(i)
        
        samples_to_process = filtered_samples
        
        if len(samples_to_process) == 0:
            return
        
        # Inference
        with torch.no_grad():
            pred_06m = self.model(image, continuous, nlcd_idx, ecoregion_idx)
            pred_06m = pred_06m.squeeze(1)
            pred_06m = torch.clamp(pred_06m, min=0.0)
        
        for i in samples_to_process:
            # 0.6m metrics
            metrics_06m = self.calculate_metrics(
                pred_06m[i], chm_06m[i].squeeze(0), valid_mask_06m[i]
            )
            
            # 1.0m metrics
            metrics_10m = None
            if enable_10m_evaluation:
                try:
                    # load original CHM
                    chm_path = chm_paths[i]
                    if self.base_path:
                        full_chm_path = os.path.join(self.base_path, chm_path)
                    else:
                        full_chm_path = chm_path
                    
                    with rasterio.open(full_chm_path) as src:
                        chm_10m_original = src.read(1).astype(np.float32)
                        original_shape = chm_10m_original.shape
                        
                        nodata_val = src.nodata if src.nodata is not None else 65535
                        valid_mask_10m_original = (chm_10m_original != nodata_val)
                        
                        chm_10m_original = chm_10m_original / 100.0
                        chm_10m_original = np.clip(chm_10m_original, 0, 120)
                        chm_10m_original[~valid_mask_10m_original] = 0
                    
                    # resample
                    pred_10m = F.interpolate(
                        pred_06m[i].unsqueeze(0).unsqueeze(0),
                        size=original_shape,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze()
                    pred_10m = torch.clamp(pred_10m, min=0.0)
                    
                    chm_10m = torch.from_numpy(chm_10m_original).to(self.device)
                    valid_mask_10m = torch.from_numpy(valid_mask_10m_original).to(self.device)
                    
                    metrics_10m = self.calculate_metrics(
                        pred_10m, chm_10m, valid_mask_10m
                    )
                
                except Exception as e:
                    print(f"Warning: Failed to load/resample original CHM for {chm_paths[i]}: {e}")
                    metrics_10m = None
            
            self.update_metrics(
                metrics_06m, metrics_10m, nlcd_classes[i], ecoregions[i]
            )

            # store stats
            if metrics_06m:
                self.tile_stats_06m.append({
                    'chm_path': chm_paths[i],
                    'nlcd_class': nlcd_classes[i],
                    'ecoregion': ecoregions[i],
                    **metrics_06m
                })
            
            if metrics_10m:
                self.tile_stats_10m.append({
                    'chm_path': chm_paths[i],
                    'nlcd_class': nlcd_classes[i],
                    'ecoregion': ecoregions[i],
                    **metrics_10m
                })
    
    def get_overall_results(self):
        """Get overall performance summary"""
        return {
            '0.6m': self.overall_06m.get_summary(),
            '1.0m': self.overall_10m.get_summary()
        }
    
    def get_nlcd_results(self):
        """Get NLCD class breakdown"""
        results = {}
        for nlcd_class in NLCD_CLASSES:
            if self.nlcd_06m[nlcd_class].count > 0:
                results[nlcd_class] = {
                    'name': self.nlcd_names[nlcd_class],
                    '0.6m': self.nlcd_06m[nlcd_class].get_summary(),
                    '1.0m': self.nlcd_10m[nlcd_class].get_summary()
                }
        return results
    
    def get_ecoregion_nlcd_results(self):
        """Get detailed ecoregion x NLCD breakdown for CSV"""
        results = []
        
        for ecoregion in self.ecoregion_nlcd_06m:
            for nlcd_class in self.ecoregion_nlcd_06m[ecoregion]:
                acc_06m = self.ecoregion_nlcd_06m[ecoregion][nlcd_class]
                acc_10m = self.ecoregion_nlcd_10m[ecoregion][nlcd_class]
                
                if acc_06m.count > 0:
                    summary_06m = acc_06m.get_summary()
                    summary_10m = acc_10m.get_summary()
                    
                    results.append({
                        'ecoregion': ecoregion,
                        'nlcd_class': nlcd_class,
                        'nlcd_name': self.nlcd_names.get(nlcd_class, f'Class_{nlcd_class}'),
                        'sample_count': summary_06m['count'],
                        
                        # 0.6m metrics
                        'mae_06m': summary_06m['mae'],
                        'rmse_06m': summary_06m['rmse'],
                        'bias_06m': summary_06m['bias'],
                        'r2_06m': summary_06m['r2'],
                        'jsd_06m': summary_06m['jsd'],
                        'avg_pred_height_06m': summary_06m['avg_pred_height'],
                        'avg_obs_height_06m': summary_06m['avg_obs_height'],
                        'avg_pred_height_p95_06m': summary_06m['avg_pred_height_p95'],
                        'avg_obs_height_p95_06m': summary_06m['avg_obs_height_p95'],
                        
                        # 1.0m metrics  
                        'mae_10m': summary_10m['mae'],
                        'rmse_10m': summary_10m['rmse'],
                        'bias_10m': summary_10m['bias'],
                        'r2_10m': summary_10m['r2'],
                        'jsd_10m': summary_10m['jsd']
                    })
        
        return results

    def merge(self, other_evaluator):
        """Merge another evaluator's metrics into this one."""
        self.overall_06m.merge(other_evaluator.overall_06m)
        self.overall_10m.merge(other_evaluator.overall_10m)

        for cls in NLCD_CLASSES:
            self.nlcd_06m[cls].merge(other_evaluator.nlcd_06m[cls])
            self.nlcd_10m[cls].merge(other_evaluator.nlcd_10m[cls])

        for ecoregion, nlcd_metrics in other_evaluator.ecoregion_nlcd_06m.items():
            for nlcd_class, accumulator in nlcd_metrics.items():
                self.ecoregion_nlcd_06m[ecoregion][nlcd_class].merge(accumulator)

        for ecoregion, nlcd_metrics in other_evaluator.ecoregion_nlcd_10m.items():
            for nlcd_class, accumulator in nlcd_metrics.items():
                self.ecoregion_nlcd_10m[ecoregion][nlcd_class].merge(accumulator)
        
        self.tile_stats_06m.extend(other_evaluator.tile_stats_06m)
        self.tile_stats_10m.extend(other_evaluator.tile_stats_10m)


class MetricAccumulator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.count = 0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.bias_sum = 0.0
        self.ss_res_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.total_pred_sum = 0.0
        self.total_target_sum = 0.0
        self.avg_pred_height_sum = 0.0
        self.avg_obs_height_sum = 0.0
        self.pred_height_p95_sum = 0.0
        self.obs_height_p95_sum = 0.0
        self.n_pixels_sum = 0
        self.jsd_sum = 0.0
        self.jsd_count = 0  # Track valid (non-NaN) JSD values
    
    def update(self, metrics):
        if metrics is None:
            return
        
        self.count += 1
        self.mae_sum += metrics['mae']
        self.rmse_sum += metrics['rmse']
        self.bias_sum += metrics['bias']
        self.ss_res_sum += metrics['ss_res']
        self.target_sum += metrics['target_sum']
        self.target_sq_sum += metrics['target_sq_sum']
        self.total_pred_sum += metrics['total_pred']
        self.total_target_sum += metrics['total_target']
        self.avg_pred_height_sum += metrics['avg_pred_height']
        self.avg_obs_height_sum += metrics['avg_obs_height']
        self.pred_height_p95_sum += metrics['pred_height_p95']
        self.obs_height_p95_sum += metrics['obs_height_p95']
        self.n_pixels_sum += metrics['n_pixels']
        
        if not np.isnan(metrics['jsd']):
            self.jsd_sum += metrics['jsd']
            self.jsd_count += 1

    def merge(self, other):
        self.count += other.count
        self.mae_sum += other.mae_sum
        self.rmse_sum += other.rmse_sum
        self.bias_sum += other.bias_sum
        self.ss_res_sum += other.ss_res_sum
        self.target_sum += other.target_sum
        self.target_sq_sum += other.target_sq_sum
        self.total_pred_sum += other.total_pred_sum
        self.total_target_sum += other.total_target_sum
        self.avg_pred_height_sum += other.avg_pred_height_sum
        self.avg_obs_height_sum += other.avg_obs_height_sum
        self.pred_height_p95_sum += other.pred_height_p95_sum
        self.obs_height_p95_sum += other.obs_height_p95_sum
        self.n_pixels_sum += other.n_pixels_sum
        self.jsd_sum += other.jsd_sum
        self.jsd_count += other.jsd_count
    
    def get_summary(self):
        if self.count == 0:
            return {
                'count': 0, 'mae': 0, 'rmse': 0, 'bias': 0, 'r2': 0, 'jsd': np.nan,
                'avg_pred_height': 0, 'avg_obs_height': 0,
                'avg_pred_height_p95': 0, 'avg_obs_height_p95': 0
            }
        
        # Aggregated R2
        if self.n_pixels_sum > 1:
            ss_tot = self.target_sq_sum - (self.target_sum ** 2) / self.n_pixels_sum
            r2 = 1 - (self.ss_res_sum / ss_tot) if ss_tot > 0 else 0.0
        else:
            r2 = 0.0
        
        # JSD avg
        if self.jsd_count > 0:
            jsd_avg = self.jsd_sum / self.jsd_count
        else:
            jsd_avg = np.nan
            
        return {
            'count': self.count,
            'mae': self.mae_sum / self.count,
            'rmse': self.rmse_sum / self.count,
            'bias': self.bias_sum / self.count,
            'r2': r2,
            'jsd': jsd_avg,
            'avg_pred_height': self.avg_pred_height_sum / self.count,
            'avg_obs_height': self.avg_obs_height_sum / self.count,
            'avg_pred_height_p95': self.pred_height_p95_sum / self.count,
            'avg_obs_height_p95': self.obs_height_p95_sum / self.count,
            'total_pixels': self.n_pixels_sum
        }


def load_model_from_checkpoint(checkpoint_path, config, device):
    print(f"Loading model from: {checkpoint_path}")
    
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('module.') for key in state_dict.keys()):
        # remove DDP prefix
        state_dict = {key.replace('module.', ''): value 
                     for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_mae' in checkpoint:
        print(f"Checkpoint validation MAE: {checkpoint['val_mae']:.3f}")
    
    return model


def create_evaluation_dataset(config, partition, metadata_file_override=None):
    print(f"Creating {partition} dataset...")
    
    # load metadata
    if metadata_file_override:
        metadata_path = metadata_file_override
        print(f"Using metadata file override: {metadata_path}")
    else:
        metadata_path = config['data'].get('metadata_path')
        if not metadata_path:
            metadata_path = config['data'].get('csv_path')
        print(f"Using metadata file from config: {metadata_path}")
    
    dataset = CanopyHeightDataset(
        metadata_path=metadata_path,
        base_path=config['data']['base_path'],
        partition=partition,
        augment=False,
        target_size=config['data']['target_size']
    )
    
    print(f"Dataset created: {len(dataset)} samples")
    return dataset


def print_overall_results(results):
    """Print overall performance summary"""
    print("\n" + "="*60)
    print("Model Evaluation Results")
    print("="*60)
    
    r06 = results['0.6m']
    r10 = results['1.0m']
    
    print(f"\nOverall Performance (0.6m Resolution):")
    print(f"  MAE: {r06['mae']:.3f} m")
    print(f"  RMSE: {r06['rmse']:.3f} m")
    print(f"  Bias: {r06['bias']:.3f} m")
    print(f"  R²: {r06['r2']:.3f}")
    print(f"  JSD: {r06['jsd']:.4f}")
    print(f"  Mean Height (Pred/Ref): {r06['avg_pred_height']:.2f}m / {r06['avg_obs_height']:.2f}m")
    print(f"  95th Pct Height (Pred/Ref): {r06['avg_pred_height_p95']:.2f}m / {r06['avg_obs_height_p95']:.2f}m")

    print(f"\nOverall Performance (1.0m Resolution):")
    print(f"  MAE: {r10['mae']:.3f} m")
    print(f"  RMSE: {r10['rmse']:.3f} m")
    print(f"  Bias: {r10['bias']:.3f} m")
    print(f"  R²: {r10['r2']:.3f}")
    print(f"  JSD: {r10['jsd']:.4f}")


def print_nlcd_results(nlcd_results):
    """Print NLCD class breakdown"""
    print(f"\nNLCD Class Performance (0.6m Resolution):")
    print(f"{'Class':>5} {'Name':>20} {'Samples':>10} {'MAE':>8} {'R²':>8} {'JSD':>8}")
    print("-" * 70)
    
    for nlcd_class in sorted(nlcd_results.keys()):
        result = nlcd_results[nlcd_class]
        r06 = result['0.6m']
        
        print(f"{nlcd_class:>5} "
              f"{result['name']:>20} "
              f"{r06['count']:>10,} "
              f"{r06['mae']:>8.2f} "
              f"{r06['r2']:>8.3f} "
              f"{r06['jsd']:>8.4f}")


def save_detailed_csv(ecoregion_results, output_path):
    """Save detailed ecoregion x NLCD results to CSV"""
    if not ecoregion_results:
        print("No ecoregion results to save")
        return
    
    df = pd.DataFrame(ecoregion_results)
    df = df.sort_values(['ecoregion', 'nlcd_class'])
    
    df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"\nDetailed results saved to: {output_path}")
    print(f"CSV contains {len(df)} ecoregion x NLCD combinations")


def save_tiledata_csv(tile_stats, metadata_df, output_path, resolution='0.6m'):
    if not tile_stats:
        print(f"No tile-level statistics to save for {resolution}.")
        return

    stats_df = pd.DataFrame(tile_stats)
    
    # merge with metadata
    merged_df = pd.merge(stats_df, metadata_df, on='chm_path', how='left')
    
    # handle duplicates
    if 'nlcd_class_x' in merged_df.columns:
        merged_df['nlcd_class'] = merged_df['nlcd_class_x']
        merged_df = merged_df.drop(columns=['nlcd_class_x', 'nlcd_class_y'])
    
    # coords
    transformer = CoordinateTransformer()
    coords_4326 = transformer.transform_batch(
        merged_df['utm_zone'].values,
        merged_df['x'].values,
        merged_df['y'].values,
        target_crs="EPSG:4326"
    )
    merged_df['lon'] = coords_4326[:, 0]
    merged_df['lat'] = coords_4326[:, 1]
    
    # Select and reorder columns - include ecoregion for aggregation
    output_cols = [
        'chm_path', 'lon', 'lat', 'utm_zone', 'nlcd_class', 'ecoregion', 'us_l3code', 'doy', 'elevation',
        'mae', 'rmse', 'bias', 'ss_res', 'ss_tot', 'target_sum', 'target_sq_sum',
        'avg_pred_height', 'avg_obs_height', 
        'pred_height_p95', 'obs_height_p95', 'n_pixels', 'jsd'
    ]
    final_df = merged_df[output_cols]
    
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nTile-level statistics ({resolution}) saved to: {output_path}")
    print(f"  {len(final_df)} tiles")


def aggregate_tiledata_to_ecoregion(tiledata_06m_path, tiledata_10m_path, output_path):
    """Aggregate tile-level data to ecoregion x NLCD combinations"""
    print("\nAggregating tiledata to ecoregion level...")
    
    # Load both tiledata files
    df_06m = pd.read_csv(tiledata_06m_path)
    df_10m = pd.read_csv(tiledata_10m_path)
    
    # NLCD class names
    nlcd_names = {
        11: 'Water', 21: 'Developed', 31: 'Barren', 41: 'Forest',
        52: 'Shrub/Scrub', 71: 'Grassland', 81: 'Pasture/Hay',
        82: 'Cultivated', 90: 'Wetlands'
    }
    
    # Group by ecoregion and NLCD class
    results = []
    
    for (ecoregion, nlcd_class), group_06m in df_06m.groupby(['ecoregion', 'nlcd_class']):
        # Get corresponding 1.0m data
        group_10m = df_10m[(df_10m['ecoregion'] == ecoregion) & (df_10m['nlcd_class'] == nlcd_class)]
        
        if len(group_06m) == 0:
            continue
        
        # Aggregate 0.6m metrics
        n_samples = len(group_06m)
        mae_06m = group_06m['mae'].mean()
        rmse_06m = group_06m['rmse'].mean()
        bias_06m = group_06m['bias'].mean()
        
        # Calculate aggregated R² for 0.6m
        total_pixels_06m = group_06m['n_pixels'].sum()
        ss_res_06m = group_06m['ss_res'].sum()
        target_sum_06m = group_06m['target_sum'].sum()
        target_sq_sum_06m = group_06m['target_sq_sum'].sum()
        
        if total_pixels_06m > 1:
            ss_tot_06m = target_sq_sum_06m - (target_sum_06m ** 2) / total_pixels_06m
            r2_06m = 1 - (ss_res_06m / ss_tot_06m) if ss_tot_06m > 0 else 0.0
        else:
            r2_06m = 0.0
        
        avg_pred_height_06m = group_06m['avg_pred_height'].mean()
        avg_obs_height_06m = group_06m['avg_obs_height'].mean()
        avg_pred_height_p95_06m = group_06m['pred_height_p95'].mean()
        avg_obs_height_p95_06m = group_06m['obs_height_p95'].mean()
        
        # Aggregate 1.0m metrics
        if len(group_10m) > 0:
            mae_10m = group_10m['mae'].mean()
            rmse_10m = group_10m['rmse'].mean()
            bias_10m = group_10m['bias'].mean()
            
            # Calculate aggregated R² for 1.0m
            total_pixels_10m = group_10m['n_pixels'].sum()
            ss_res_10m = group_10m['ss_res'].sum()
            target_sum_10m = group_10m['target_sum'].sum()
            target_sq_sum_10m = group_10m['target_sq_sum'].sum()
            
            if total_pixels_10m > 1:
                ss_tot_10m = target_sq_sum_10m - (target_sum_10m ** 2) / total_pixels_10m
                r2_10m = 1 - (ss_res_10m / ss_tot_10m) if ss_tot_10m > 0 else 0.0
            else:
                r2_10m = 0.0
        else:
            mae_10m = rmse_10m = bias_10m = r2_10m = 0.0
        
        results.append({
            'ecoregion': ecoregion,
            'nlcd_class': nlcd_class,
            'nlcd_name': nlcd_names.get(nlcd_class, f'Class_{nlcd_class}'),
            'sample_count': n_samples,
            'mae_06m': mae_06m,
            'rmse_06m': rmse_06m,
            'bias_06m': bias_06m,
            'r2_06m': r2_06m,
            'avg_pred_height_06m': avg_pred_height_06m,
            'avg_obs_height_06m': avg_obs_height_06m,
            'avg_pred_height_p95_06m': avg_pred_height_p95_06m,
            'avg_obs_height_p95_06m': avg_obs_height_p95_06m,
            'mae_10m': mae_10m,
            'rmse_10m': rmse_10m,
            'bias_10m': bias_10m,
            'r2_10m': r2_10m,
        })
    
    # Save aggregated results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(['ecoregion', 'nlcd_class'])
    df_results.to_csv(output_path, index=False, float_format='%.3f')
    
    print(f"Ecoregion aggregation saved to: {output_path}")
    print(f"  {len(df_results)} ecoregion x NLCD combinations")


def generate_markdown_report(overall_results, nlcd_results, ecoregion_results, 
                           checkpoint_path, config_path, partition, output_path):
    """Generate comprehensive markdown evaluation report"""
    
    from datetime import datetime
    
    # Extract checkpoint info
    checkpoint_name = Path(checkpoint_path).stem
    
    # Build markdown content
    markdown = []
    
    # Header
    markdown.extend([
        f"# Canopy Height Model Evaluation Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Checkpoint:** `{checkpoint_name}`  ",
        f"**Config:** `{Path(config_path).name}`  ",
        f"**Partition:** {partition}  ",
        f"",
        f"---",
        f""
    ])
    
    # Overall Performance Summary
    r06 = overall_results['0.6m']
    r10 = overall_results['1.0m']
    
    markdown.extend([
        f"## Overall Performance",
        f"",
        f"| Metric | 0.6m Resolution | 1.0m Resolution |",
        f"|--------|-----------------|-----------------|",
        f"| **MAE (m)** | {r06['mae']:.3f} | {r10['mae']:.3f} |",
        f"| **RMSE (m)** | {r06['rmse']:.3f} | {r10['rmse']:.3f} |",
        f"| **Bias (m)** | {r06['bias']:.3f} | {r10['bias']:.3f} |",
        f"| **R²** | {r06['r2']:.3f} | {r10['r2']:.3f} |",
        f"| **JSD** | {r06['jsd']:.4f} | {r10['jsd']:.4f} |",
        f"| **Sample Count** | {r06['count']:,} | {r10['count']:,} |",
        f"",
        f"### Key Findings",
        f"",
    ])
        
    markdown.extend([f"", f"---", f""])
    
    # NLCD Class Performance
    markdown.extend([
        f"## Performance by Land Cover Class",
        f"",
        f"| NLCD | Class Name | Samples | 0.6m MAE | 0.6m R² | 0.6m JSD | 1.0m MAE | 1.0m R² | 1.0m JSD |",
        f"|------|------------|---------|----------|---------|----------|----------|---------|----------|"
    ])
    
    for nlcd_class in sorted(nlcd_results.keys()):
        result = nlcd_results[nlcd_class]
        r06_class = result['0.6m']
        r10_class = result['1.0m']
        
        markdown.append(
            f"| {nlcd_class} | {result['name']} | {r06_class['count']:,} | "
            f"{r06_class['mae']:.2f} | {r06_class['r2']:.3f} | {r06_class['jsd']:.4f} | "
            f"{r10_class['mae']:.2f} | {r10_class['r2']:.3f} | {r10_class['jsd']:.4f} |"
        )
      
    # Ecoregion Analysis Summary
    if ecoregion_results:
        markdown.extend([
            f"## Ecoregion Performance Summary",
            f"",
            f"**Total ecoregion x land cover combinations:** {len(ecoregion_results)}",
            f""
        ])
        
        # Ecoregion performance summary
        ecoregion_summary = {}
        for result in ecoregion_results:
            eco = result['ecoregion']
            if eco not in ecoregion_summary:
                ecoregion_summary[eco] = {'mae_06m': [], 'samples': 0}
            ecoregion_summary[eco]['mae_06m'].append(result['mae_06m'])
            ecoregion_summary[eco]['samples'] += result['sample_count']
        
        # Calculate average MAE per ecoregion
        ecoregion_avg = [(np.mean(data['mae_06m']), eco, data['samples']) 
                        for eco, data in ecoregion_summary.items()]
        ecoregion_avg.sort()
        
        markdown.extend([
            f"| Ecoregion | Avg MAE (m) | Total Samples |",
            f"|-----------|-------------|---------------|"
        ])
        
        for mae, eco, samples in ecoregion_avg[:10]:  # Top 10 best performing
            markdown.append(f"| {eco} | {mae:.2f} | {samples:,} |")
        
        if len(ecoregion_avg) > 10:
            markdown.append(f"| ... | ... | ... |")
            markdown.append(f"| *(showing top 10 ecoregions)* | | |")
    
    markdown.extend([f"", f"---", f""])
    
    # Technical Details
    markdown.extend([
        f"## Technical Details",
        f"",
        f"### Evaluation Methodology",
        f"- **Model resolution**: 0.6m native, 1.0m resampled",
        f"- **Metrics calculated**: MAE, RMSE, Bias, R², JSD (Jensen-Shannon Divergence)",
        f"- **JSD parameters**: 60 bins spanning 0-60m (1m bin width)",
        f"- **Resampling method**: Nearest neighbor interpolation for 1.0m evaluation",
        f"",
        f"### Data Coverage",
        f"- **Validation samples**: {r06['count']:,}",
        f"- **Total pixels evaluated**: {r06['total_pixels']:,}",
        f"- **Land cover classes**: {len(nlcd_results)} NLCD classes represented",
        f"",
        f"### File Outputs",
        f"- **Detailed CSV**: Contains all ecoregion x NLCD combinations",
        f"- **Console output**: Summary statistics and performance breakdown",
        f"",
        f"---",
        f"",
        f"*Report generated by the Canopy Height Model evaluation pipeline*"
    ])
    
    # Write markdown file
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"\nMarkdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate canopy height model")
    
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', required=True,
                       help='Path to config file')
    parser.add_argument('--partition', default='validate',
                       choices=['validate', 'test'],
                       help='Dataset partition to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loader workers')
    parser.add_argument('--output_csv', 
                       help='Path to save detailed CSV results')
    parser.add_argument('--output_tile_stats',
                       help='Path to save tile-level statistics CSV')
    parser.add_argument('--output_markdown', 
                       help='Path to save markdown summary report')
    parser.add_argument('--metadata_file',
                       help='Path to metadata parquet file (overrides config metadata_path)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to randomly select for evaluation')
    
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_ddp = world_size > 1

    if is_ddp:
        setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    if is_ddp:
        model = DDP(model, device_ids=[rank])
    
    # Create dataset and dataloader
    dataset = create_evaluation_dataset(config, args.partition, args.metadata_file)

    metadata_df = None
    if args.output_tile_stats:
        print("Loading metadata for tile-level statistics...")
        metadata_path = args.metadata_file or config['data'].get('metadata_path')
        cols_to_load = ['chm', 'x', 'y', 'utm_zone', 'land_cover', 'us_l3code', 'doy', 'elevation']
        metadata_df = pd.read_parquet(metadata_path, columns=cols_to_load)
        metadata_df = metadata_df.rename(columns={'chm': 'chm_path', 'land_cover': 'nlcd_class'})
    
    if args.num_samples is not None and args.num_samples < len(dataset):
        if rank == 0:
            print(f"Randomly selecting {args.num_samples} samples for evaluation...")
        
        # Create a subset of the dataset
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    
    batch_size_per_gpu = args.batch_size // world_size if is_ddp else args.batch_size
    num_workers_per_gpu = args.num_workers // world_size if is_ddp else args.num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=num_workers_per_gpu,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False,
        sampler=sampler
    )
    
    # init evaluator
    evaluator = ModelEvaluator(
        model, 
        device, 
        config['data']['target_size'],
        base_path=config['data']['base_path']
    )
    
    if rank == 0:
        print(f"\nEvaluating {len(dataset)} samples in {len(dataloader)} batches...")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating (Rank {rank})", disable=(rank!=0))):
            evaluator.evaluate_batch(batch, enable_10m_evaluation=True)
            
            if rank == 0 and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                samples_processed = (batch_idx + 1) * args.batch_size
                rate = samples_processed / elapsed
                print(f"Processed {samples_processed:,} samples ({rate:.1f} samples/sec)")

    if is_ddp:
        # Gather evaluators from all processes
        gathered_evaluators = [None] * world_size
        dist.gather_object(evaluator, gathered_evaluators if rank == 0 else None, dst=0)

        if rank == 0:
            print("Merging results from all GPUs...")
            main_evaluator = gathered_evaluators[0]
            for i in range(1, world_size):
                main_evaluator.merge(gathered_evaluators[i])
            evaluator = main_evaluator

    if rank == 0:
        # Get results
        overall_results = evaluator.get_overall_results()
        nlcd_results = evaluator.get_nlcd_results()
        ecoregion_results = evaluator.get_ecoregion_nlcd_results()
        
        # Print results
        print_overall_results(overall_results)
        print_nlcd_results(nlcd_results)
        
        # Save tile-level data for both resolutions
        if args.output_tile_stats:
            # Determine output paths for both resolutions
            base_path = Path(args.output_tile_stats)
            output_dir = base_path.parent
            base_name = base_path.stem
            
            tiledata_06m_path = output_dir / f"{base_name}_06m.csv"
            tiledata_10m_path = output_dir / f"{base_name}_10m.csv"
            
            # Save separate tiledata files
            save_tiledata_csv(evaluator.tile_stats_06m, metadata_df, tiledata_06m_path, resolution='0.6m')
            save_tiledata_csv(evaluator.tile_stats_10m, metadata_df, tiledata_10m_path, resolution='1.0m')
            
            # Aggregate to ecoregion level if output_csv is specified
            if args.output_csv:
                aggregate_tiledata_to_ecoregion(tiledata_06m_path, tiledata_10m_path, args.output_csv)
        elif args.output_csv:
            # Fallback: save ecoregion results directly from accumulators
            save_detailed_csv(ecoregion_results, args.output_csv)
        
        # Generate markdown report
        if args.output_markdown:
            generate_markdown_report(
                overall_results, nlcd_results, ecoregion_results,
                args.checkpoint, args.config, args.partition, args.output_markdown
            )
        
        # Final summary
        elapsed_total = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_total/60:.1f} minutes")
        print(f"Average rate: {len(dataset)/elapsed_total:.1f} samples/second")

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
