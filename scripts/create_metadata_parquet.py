"""
Create unified metadata Parquet file.
Extracts elevation and climate/soil PCA features.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from coordinate_transform import CoordinateTransformer
from raster_sampling import RasterSampler


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('metadata_extraction.log')
        ]
    )
    return logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe memory usage."""
    logger = logging.getLogger(__name__)
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize ints
    int_cols = ['utm_zone', 'land_cover', 'us_l3code', 'doy']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
    
    # Optimize floats
    float_cols = ['x', 'y', 'elevation'] + [f'climate_pca_{i}' for i in range(1, 7)] + [f'soil_pca_{i}' for i in range(1, 7)]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    # Optimize strings
    string_cols = ['chm', 'naip', 'partition']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - optimized_memory / original_memory) * 100
    
    logger.info(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                f"({reduction:.1f}% reduction)")
    
    return df


def validate_input_data(csv_path: str, base_path: str) -> pd.DataFrame:
    """Load and validate input CSV."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading CSV from {csv_path}")
    
    # Check file existence
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} samples from CSV")
    
    # Check required columns
    required_cols = ['chm', 'naip', 'utm_zone', 'x', 'y', 'land_cover', 'doy', 'partition']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check UTM zones
    utm_zones = df['utm_zone'].unique()
    logger.info(f"UTM zones in data: {sorted(utm_zones)}")
    
    invalid_zones = [zone for zone in utm_zones if not (10 <= zone <= 19)]
    if invalid_zones:
        logger.warning(f"Unexpected UTM zones found: {invalid_zones}")
    
    # Check partitions
    partitions = df['partition'].value_counts()
    logger.info(f"Data partitions: {dict(partitions)}")
    
    # Check missing values
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.any():
        logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
    
    return df


def extract_auxiliary_features(df: pd.DataFrame,
                              elevation_path: str,
                              pca_path: str,
                              soil_pca_path: str,
                              chunk_size: int = 50000) -> pd.DataFrame:
    """Extract elevation, climate PCA, and soil PCA features."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting auxiliary feature extraction")
    
    # Init transformer
    coord_transformer = CoordinateTransformer()
    
    # Init result columns
    df = df.copy()
    df['elevation'] = np.nan
    for i in range(1, 7):
        df[f'climate_pca_{i}'] = np.nan
    for i in range(1, 7):
        df[f'soil_pca_{i}'] = np.nan
    
    n_samples = len(df)
    processed = 0
    failed = 0
    
    # Process chunks
    with RasterSampler(elevation_path, pca_path, soil_pca_path) as sampler:
        
        # Get raster info
        raster_info = sampler.get_raster_info()
        logger.info(f"Raster info: {raster_info}")
        
        # Process chunks
        with tqdm(total=n_samples, desc="Extracting features") as pbar:
            
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk = df.iloc[chunk_start:chunk_end]
                
                # Transform coordinates
                coords_5070 = coord_transformer.transform_batch(
                    chunk['utm_zone'].values,
                    chunk['x'].values,
                    chunk['y'].values
                )
                
                # Sample features
                elevations, pca_values_list, soil_pca_values_list = sampler.sample_batch(coords_5070)
                
                # Save results
                for i, (idx, (elevation, pca_values, soil_pca_values)) in enumerate(zip(chunk.index, zip(elevations, pca_values_list, soil_pca_values_list))):
                    if elevation is not None and pca_values is not None and soil_pca_values is not None:
                        df.loc[idx, 'elevation'] = elevation
                        for j, pca_val in enumerate(pca_values):
                            df.loc[idx, f'climate_pca_{j+1}'] = pca_val
                        for j, soil_pca_val in enumerate(soil_pca_values):
                            df.loc[idx, f'soil_pca_{j+1}'] = soil_pca_val
                        processed += 1
                    else:
                        failed += 1
                
                pbar.update(len(chunk))
                
                # Log status
                if (chunk_start // chunk_size + 1) % 10 == 0:
                    success_rate = processed / (processed + failed) * 100
                    logger.info(f"Processed {chunk_end:,}/{n_samples:,} samples "
                               f"(Success: {success_rate:.1f}%)")
    
    # Log stats
    total_processed = processed + failed
    success_rate = processed / total_processed * 100 if total_processed > 0 else 0
    
    logger.info(f"Feature extraction completed:")
    logger.info(f"  Total samples: {n_samples:,}")
    logger.info(f"  Successfully processed: {processed:,} ({success_rate:.1f}%)")
    logger.info(f"  Failed: {failed:,}")
    
    # Check missing features
    missing_elevation = df['elevation'].isnull().sum()
    missing_pca = df[[f'climate_pca_{i}' for i in range(1, 7)]].isnull().any(axis=1).sum()
    missing_soil_pca = df[[f'soil_pca_{i}' for i in range(1, 7)]].isnull().any(axis=1).sum()
    
    if missing_elevation > 0:
        logger.warning(f"Missing elevation values: {missing_elevation:,}")
    if missing_pca > 0:
        logger.warning(f"Missing PCA values: {missing_pca:,}")
    if missing_soil_pca > 0:
        logger.warning(f"Missing soil PCA values: {missing_soil_pca:,}")
    
    return df


def save_metadata_parquet(df: pd.DataFrame, output_path: str, 
                         partition_cols: Optional[list] = None) -> None:
    """Save metadata as optimized Parquet file."""
    logger = logging.getLogger(__name__)
    
    # Optimize dtypes
    df = optimize_dtypes(df)
    
    # Ensure output dir
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving metadata to {output_path}")
    
    # Save with compression
    if partition_cols:
        # Partitioned write
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            partition_cols=partition_cols,
            index=False
        )
        logger.info(f"Saved partitioned Parquet by {partition_cols}")
    else:
        # Single file write
        df.to_parquet(
            output_path,
            engine='pyarrow', 
            compression='snappy',
            index=False
        )
        logger.info(f"Saved single Parquet file")
    
    # Log size
    if output_path.is_file():
        file_size = output_path.stat().st_size / 1024**2
        logger.info(f"Output file size: {file_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Create unified metadata Parquet file")
    
    parser.add_argument('--csv_path', 
                       default='/local-scratch/smorf/naip-height-map/data/files.csv',
                       help='Path to input CSV file')
    
    parser.add_argument('--base_path', 
                       default='/local-scratch/smorf/naip-height-map/data',
                       help='Base path for data files')
    
    parser.add_argument('--elevation_path',
                       default='/local-scratch/smorf/naip-height-map/data/auxiliary/elevation_srtm_240m_5070.tif',
                       help='Path to elevation COG file')
    
    parser.add_argument('--pca_path',
                       default='/local-scratch/smorf/naip-height-map/data/auxiliary/climate_pca_5070.tif',
                       help='Path to climate PCA COG file')

    parser.add_argument('--soil_pca_path',
                          default='/local-scratch/smorf/naip-height-map/data/auxiliary/soil_pca_5070.tif',
                          help='Path to soil PCA COG file')
    
    parser.add_argument('--output_path',
                       default='/local-scratch/smorf/naip-height-map/data/metadata.parquet',
                       help='Output Parquet file path')
    
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='Chunk size for processing')
    
    parser.add_argument('--partition', action='store_true',
                       help='Create partitioned Parquet by data partition')
    
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Init logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting metadata Parquet creation")
    logger.info(f"Configuration:")
    logger.info(f"  CSV path: {args.csv_path}")
    logger.info(f"  Elevation COG: {args.elevation_path}")
    logger.info(f"  PCA COG: {args.pca_path}")
    logger.info(f"  Soil PCA COG: {args.soil_pca_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info(f"  Chunk size: {args.chunk_size:,}")
    
    start_time = time.time()
    
    try:
        # Validate inputs
        df = validate_input_data(args.csv_path, args.base_path)
        
        # Extract features
        df = extract_auxiliary_features(
            df, 
            args.elevation_path, 
            args.pca_path,
            args.soil_pca_path,
            chunk_size=args.chunk_size
        )
        
        # Save results
        partition_cols = ['partition'] if args.partition else None
        save_metadata_parquet(df, args.output_path, partition_cols)
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully created metadata Parquet file")
        logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Processing rate: {len(df)/elapsed_time:.1f} samples/second")
        
    except Exception as e:
        logger.error(f"Failed to create metadata Parquet: {e}")
        raise


if __name__ == "__main__":
    main()
