"""
Canopy Height Model Dataset.
Loading, preprocessing, augmentation.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path


# NLCD class mapping
NLCD_CLASSES = [11, 21, 31, 41, 52, 71, 81, 82, 90]
NLCD_TO_IDX = {cls: idx for idx, cls in enumerate(NLCD_CLASSES)}

# Column name mapping for backward compatibility
COLUMN_MAPPING = {
    'chm': 'chm_path',
    'naip': 'naip_path', 
    'land_cover': 'nlcd_class'
}

# PCA standardization constants
PCA_STD_DEVS = [
    3.0962563,   # climate_pca_1
    2.518362,    # climate_pca_2  
    1.0967529,   # climate_pca_3
    0.9703195,   # climate_pca_4
    0.60237694,  # climate_pca_5
    0.52861726   # climate_pca_6
]

# Soil PCA standardization constants
SOIL_PCA_STD_DEVS = [
    2.1590624,   # soil_pca_1
    1.7561878,   # soil_pca_2
    1.7615303,   # soil_pca_3
    1.382909,    # soil_pca_4
    1.2904456,   # soil_pca_5
    1.0198739    # soil_pca_6
]


class CanopyHeightDataset(Dataset):
    """Dataset for CHM prediction."""
    
    def __init__(self, metadata_path=None, base_path=None, partition='train', 
                 augment=False, target_size=432, metadata_df=None):
        """
        Args:
            metadata_path: Path to metadata (Parquet/CSV).
            base_path: Base path for data.
            partition: 'train', 'validate', or 'test'.
            augment: Apply augmentations.
            target_size: Target image size.
            metadata_df: Pre-loaded DataFrame.
        """
        self.base_path = base_path
        self.partition = partition
        self.augment = augment and (partition == 'train')
        self.target_size = target_size
        
        # Load data
        if metadata_df is not None:
            self.df = metadata_df.copy()
            print(f"Using pre-loaded DataFrame: {len(self.df)} samples for {partition}")
        else:
            if metadata_path is None:
                raise ValueError("Either metadata_path or metadata_df must be provided")
            self.df = self._load_metadata(metadata_path, partition)
            print(f"Loaded from file: {len(self.df)} samples for {partition}")
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for partition '{partition}'")
        
        # Check features
        self._validate_features()
        
        # Init augmentation
        if self.augment:
            self.augmentation = SpectralAugmentation()
    
    def _load_metadata(self, metadata_path: str, partition: str) -> pd.DataFrame:
        """Load metadata (Parquet/CSV)."""
        metadata_path = Path(metadata_path)
        
        if metadata_path.suffix.lower() == '.parquet':
            if metadata_path.is_dir():
                # Partitioned directory
                partition_path = metadata_path / f"partition={partition}"
                if partition_path.exists():
                    df = pd.read_parquet(partition_path)
                else:
                    df = pd.read_parquet(metadata_path)
                    df = df[df['partition'] == partition]
            else:
                # Single file
                df = pd.read_parquet(
                    metadata_path,
                    filters=[('partition', '==', partition)]
                )
            print(f"Loaded from Parquet: {metadata_path}")
            
        elif metadata_path.suffix.lower() == '.csv':
            # CSV
            df = pd.read_csv(metadata_path)
            df = df[df['partition'] == partition]
            print(f"Loaded from CSV: {metadata_path}")
            
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")
        
        # Column mapping
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _validate_features(self):
        """Check required features."""
        required_columns = [
            'chm_path', 'naip_path', 'nlcd_class', 'doy', 'elevation', 'us_l3code'
        ] + [f'climate_pca_{i}' for i in range(1, 7)] + [f'soil_pca_{i}' for i in range(1, 7)]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter missing
        feature_columns = ['elevation'] + [f'climate_pca_{i}' for i in range(1, 7)] + [f'soil_pca_{i}' for i in range(1, 7)]
        initial_count = len(self.df)
        
        valid_mask = ~self.df[feature_columns].isnull().any(axis=1)
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        excluded_count = initial_count - len(self.df)
        if excluded_count > 0:
            print(f"Filtered out {excluded_count} samples with missing auxiliary features")
            print(f"Remaining samples: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load NAIP imagery and mask
        naip_path = os.path.join(self.base_path, row['naip_path'])
        image, naip_mask = self.load_naip(naip_path)
        
        # Load and resample CHM
        chm_path = os.path.join(self.base_path, row['chm_path'])
        chm, chm_valid_mask = self.load_and_resample_chm(chm_path)
        
        # Combine masks: valid only where both NAIP and CHM are valid
        valid_mask = naip_mask & chm_valid_mask
        
        # Process auxiliary features
        continuous = self.process_auxiliary_features(row)
        nlcd_idx = NLCD_TO_IDX[row['nlcd_class']]
        ecoregion_idx = row['us_l3code']
        
        # Apply augmentations if training
        if self.augment:
            image, chm, valid_mask = self.augmentation(image, chm, valid_mask)
        
        # Ensure arrays are contiguous before converting to tensors
        # (handles negative strides from augmentation flips)
        if not image.flags['C_CONTIGUOUS']:
            image = image.copy()
        if not chm.flags['C_CONTIGUOUS']:
            chm = chm.copy()
        if not valid_mask.flags['C_CONTIGUOUS']:
            valid_mask = valid_mask.copy()
        
        # Prepare return dictionary
        result = {
            'image': torch.from_numpy(image).float(),
            'chm': torch.from_numpy(chm).float().unsqueeze(0),
            'valid_mask': torch.from_numpy(valid_mask).bool(),
            'continuous': torch.from_numpy(continuous).float(),
            'nlcd_idx': torch.tensor(nlcd_idx).long(),
            'ecoregion_idx': torch.tensor(ecoregion_idx).long(),
            'chm_path': row['chm_path']
        }
        
        return result
    
    def load_naip(self, path):
        """Load 5-band NAIP, resample to target."""
        with rasterio.open(path) as src:
            # Read all 5 bands (RGBN + mask)
            data = src.read()
            
            # Extract RGBN bands and mask
            rgbn_bands = data[:4].astype(np.float32)
            naip_mask = data[4]
            
            # Check if resampling is needed (427×427 → 432×432)
            if rgbn_bands.shape[1:] != (self.target_size, self.target_size):
                # Calculate target transform
                src_bounds = src.bounds
                dst_transform = rasterio.transform.from_bounds(
                    src_bounds.left,
                    src_bounds.bottom,
                    src_bounds.right,
                    src_bounds.top,
                    self.target_size,
                    self.target_size
                )
                
                naip_data_resampled = np.empty((5, self.target_size, self.target_size), 
                                             dtype=np.float32)
                
                # Resample RGBN
                reproject(
                    source=rasterio.band(src, [1, 2, 3, 4]),
                    destination=naip_data_resampled[:4],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                
                # Resample mask
                reproject(
                    source=rasterio.band(src, 5),
                    destination=naip_data_resampled[4],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                
                rgbn_bands = naip_data_resampled[:4].astype(np.float32)
                naip_mask = naip_data_resampled[4] > 0.5
            else:
                naip_mask = naip_mask > 0
            
            # Normalize 0-1
            rgbn_bands = rgbn_bands / 255.0
            
            return rgbn_bands, naip_mask
    
    def load_and_resample_chm(self, path):
        """Load CHM, resample to 0.6m."""
        with rasterio.open(path) as src:
            # Read CHM
            src_data = src.read(1)
            src_transform = src.transform
            src_bounds = src.bounds
            
            # Calculate transform for 0.6m
            dst_transform = rasterio.transform.from_bounds(
                src_bounds.left,
                src_bounds.bottom,
                src_bounds.right,
                src_bounds.top,
                self.target_size,
                self.target_size
            )
            
            # Resample
            dst_data = np.empty((self.target_size, self.target_size), 
                              dtype=np.float32)
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            
            # Handle NoData
            nodata_val = src.nodata if src.nodata is not None else 65535
            valid_mask = (dst_data != nodata_val)
            
            # Meters conversion and clipping
            dst_data = dst_data.astype(np.float32) / 100.0
            dst_data = np.clip(dst_data, 0, 120)
            dst_data[~valid_mask] = 0
            
            return dst_data, valid_mask
    
    def process_auxiliary_features(self, row):
        """Normalize/process aux features."""
        features = []
        
        # Climate PCA
        for i in range(1, 7):
            pca_val = row[f'climate_pca_{i}']
            normalized_pca = pca_val / PCA_STD_DEVS[i-1]
            features.append(normalized_pca)

        # Soil PCA
        for i in range(1, 7):
            soil_pca_val = row[f'soil_pca_{i}']
            normalized_soil_pca = soil_pca_val / SOIL_PCA_STD_DEVS[i-1]
            features.append(normalized_soil_pca)
        
        # Elevation
        elevation = row['elevation']
        elevation = np.clip(elevation / 4000.0, 0, 1)
        features.append(elevation)
        
        # DOY
        doy = row['doy']
        features.append(np.sin(2 * np.pi * doy / 365))
        features.append(np.cos(2 * np.pi * doy / 365))
        
        return np.array(features, dtype=np.float32)


class SpectralAugmentation:
    """Band-independent augmentation."""
    
    def __init__(self, p_spatial=0.5, p_independent=0.4, p_global=0.3):
        self.p_spatial = p_spatial
        self.p_independent = p_independent
        self.p_global = p_global
    
    def __call__(self, image, chm, valid_mask):
        # Spatial
        if random.random() < self.p_spatial:
            image, chm, valid_mask = self.apply_spatial(image, chm, valid_mask)
        
        # Independent spectral
        if random.random() < self.p_independent:
            image = self.apply_independent_band_variation(image)
        
        # Global spectral
        if random.random() < self.p_global:
            image = self.apply_global_spectral(image)
        
        return image, chm, valid_mask
    
    def apply_spatial(self, image, chm, valid_mask):
        """Spatial augmentations (rot/flip)."""
        # Random 90 deg rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            chm = np.rot90(chm, k)
            valid_mask = np.rot90(valid_mask, k)
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            chm = np.flip(chm, axis=1).copy()
            valid_mask = np.flip(valid_mask, axis=1).copy()
        
        # Random vertical flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            chm = np.flip(chm, axis=0).copy()
            valid_mask = np.flip(valid_mask, axis=0).copy()
        
        return image, chm, valid_mask
    
    def apply_independent_band_variation(self, image):
        """Independent per-band gain/offset."""
        for i in range(4):  # R, G, B, NIR
            gain = random.uniform(0.85, 1.15)
            offset = random.uniform(-0.05, 0.05)
            image[i] = np.clip(image[i] * gain + offset, 0, 1)
        
        return image
    
    def apply_global_spectral(self, image):
        """Global brightness/contrast."""
        # Brightness
        brightness = random.uniform(-0.05, 0.05)
        
        # Contrast
        contrast = random.uniform(0.9, 1.1)
        mean = image.mean()
        
        image = (image - mean) * contrast + mean + brightness
        
        return np.clip(image, 0, 1)


def create_dataloaders(config, rank=0, world_size=1):
    """Create train/val dataloaders."""
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Get metadata path
    metadata_path = config['data'].get('metadata_path')
    if not metadata_path:
        metadata_path = config['data'].get('csv_path', 'data/files.csv')
    
    print(f"Loading metadata once in main process from: {metadata_path}")
    
    # Load metadata ONCE
    metadata_path = Path(metadata_path)
    if metadata_path.suffix.lower() == '.parquet':
        if metadata_path.is_dir():
            # Partitioned Parquet
            train_path = metadata_path / "partition=train"
            val_path = metadata_path / "partition=validate"
            if train_path.exists() and val_path.exists():
                print(f"Loading partitioned Parquet from {train_path} and {val_path}")
                train_df = pd.read_parquet(train_path)
                val_df = pd.read_parquet(val_path)
            else:
                print(f"Partitioned directories not found, reading full file and filtering")
                full_df = pd.read_parquet(metadata_path)
                train_df = full_df[full_df['partition'] == 'train'].reset_index(drop=True)
                val_df = full_df[full_df['partition'] == 'validate'].reset_index(drop=True)
        else:
            # Single Parquet
            print(f"Loading filtered Parquet from {metadata_path}")
            try:
                train_df = pd.read_parquet(metadata_path, filters=[('partition', '==', 'train')])
                val_df = pd.read_parquet(metadata_path, filters=[('partition', '==', 'validate')])
            except Exception as e:
                print(f"Filtered loading failed ({e}), falling back to full load")
                full_df = pd.read_parquet(metadata_path)
                train_df = full_df[full_df['partition'] == 'train'].reset_index(drop=True)
                val_df = full_df[full_df['partition'] == 'validate'].reset_index(drop=True)
    else:
        # CSV
        full_df = pd.read_csv(metadata_path)
        train_df = full_df[full_df['partition'] == 'train'].reset_index(drop=True)
        val_df = full_df[full_df['partition'] == 'validate'].reset_index(drop=True)
    
    # Apply mapping
    train_df = train_df.rename(columns=COLUMN_MAPPING)
    val_df = val_df.rename(columns=COLUMN_MAPPING)
    
    print(f"Loaded metadata: {len(train_df)} train, {len(val_df)} validation samples")
    
    # Create datasets
    train_dataset = CanopyHeightDataset(
        base_path=config['data']['base_path'],
        partition='train',
        augment=config['data'].get('augment_train', True),
        target_size=config['data']['target_size'],
        metadata_df=train_df
    )
    
    val_dataset = CanopyHeightDataset(
        base_path=config['data']['base_path'],
        partition='validate',
        augment=False,
        target_size=config['data']['target_size'],
        metadata_df=val_df
    )
    
    # Create samplers
    train_sampler = None
    val_sampler = None
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    
    # Create dataloaders
    batch_size_per_gpu = config['training']['batch_size'] // world_size
    num_workers_per_gpu = config['dataloader']['num_workers'] // world_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers_per_gpu,
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers'],
        prefetch_factor=config['dataloader']['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers_per_gpu,
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers'],
        prefetch_factor=config['dataloader']['prefetch_factor']
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def load_metadata_info(metadata_path: str) -> dict:
    """Get metadata stats."""
    metadata_path = Path(metadata_path)
    
    if metadata_path.suffix.lower() == '.parquet':
        if metadata_path.is_dir():
            # Partitioned Parquet
            partitions = [p.name.split('=')[1] for p in metadata_path.iterdir() 
                         if p.is_dir() and p.name.startswith('partition=')]
            total_samples = 0
            for partition in partitions:
                partition_df = pd.read_parquet(metadata_path / f"partition={partition}")
                total_samples += len(partition_df)
        else:
            # Single Parquet
            df = pd.read_parquet(metadata_path)
            partitions = df['partition'].unique().tolist()
            total_samples = len(df)
    else:
        # CSV
        df = pd.read_csv(metadata_path)
        partitions = df['partition'].unique().tolist()
        total_samples = len(df)
    
    return {
        'path': str(metadata_path),
        'format': metadata_path.suffix.lower(),
        'total_samples': total_samples,
        'partitions': partitions
    }


if __name__ == "__main__":
    # Test dataset loading
    
    print("Dataset implementation ready")
    print(f"NLCD classes: {NLCD_CLASSES}")
    print(f"NLCD mapping: {NLCD_TO_IDX}")
    print(f"Column mapping: {COLUMN_MAPPING}")
    
    # Example usage:
    print("\nExample usage:")
    print("# Load from Parquet (preferred)")
    print("dataset = CanopyHeightDataset('metadata.parquet', '/data/path', 'train')")
    print("\n# Load from CSV (backward compatibility)")
    print("dataset = CanopyHeightDataset('files.csv', '/data/path', 'train')")
    
    # Test metadata info loading
    print("\nTo get metadata info:")
    print("info = load_metadata_info('metadata.parquet')")
