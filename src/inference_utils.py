"""
Inference utilities.
Persistent raster handling, optimized I/O.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform_bounds
import torch
import yaml

from .model import create_model
from .dataset import NLCD_TO_IDX, PCA_STD_DEVS, SOIL_PCA_STD_DEVS

logger = logging.getLogger(__name__)


class StaticRasterHandler:
    """Persistent raster handles."""
    
    def __init__(
        self,
        elevation_path: Path,
        climate_pca_path: Path,
        soil_pca_path: Path,
        nlcd_path: Path,
        ecoregion_path: Path
    ):
        """Open static rasters."""
        self.elevation_path = elevation_path
        self.climate_pca_path = climate_pca_path
        self.soil_pca_path = soil_pca_path
        self.nlcd_path = nlcd_path
        self.ecoregion_path = ecoregion_path
        
        # Open raster handles
        logger.info("Opening static rasters for persistent access")
        self.elevation_src = rasterio.open(elevation_path)
        self.climate_pca_src = rasterio.open(climate_pca_path)
        self.soil_pca_src = rasterio.open(soil_pca_path)
        self.nlcd_src = rasterio.open(nlcd_path)
        self.ecoregion_src = rasterio.open(ecoregion_path)
        logger.info("Static rasters opened successfully")
    
    def sample_point(self, x_5070: float, y_5070: float) -> Tuple[
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[int],
        Optional[int]
    ]:
        """
        Sample rasters at point.
        
        Args:
            x_5070: X coordinate (EPSG:5070).
            y_5070: Y coordinate (EPSG:5070).
            
        Returns:
            (elevation, climate_pca, soil_pca, nlcd, ecoregion)
        """
        try:
            # Elevation
            row, col = self.elevation_src.index(x_5070, y_5070)
            if 0 <= row < self.elevation_src.height and 0 <= col < self.elevation_src.width:
                window = Window(col, row, 1, 1)
                elevation = float(self.elevation_src.read(1, window=window)[0, 0])
            else:
                elevation = None
            
            # Climate PCA (6 bands)
            row, col = self.climate_pca_src.index(x_5070, y_5070)
            if 0 <= row < self.climate_pca_src.height and 0 <= col < self.climate_pca_src.width:
                window = Window(col, row, 1, 1)
                climate_pca = self.climate_pca_src.read(window=window)[:, 0, 0]  # (6,)
            else:
                climate_pca = None
            
            # Soil PCA (6 bands)
            row, col = self.soil_pca_src.index(x_5070, y_5070)
            if 0 <= row < self.soil_pca_src.height and 0 <= col < self.soil_pca_src.width:
                window = Window(col, row, 1, 1)
                soil_pca = self.soil_pca_src.read(window=window)[:, 0, 0]  # (6,)
            else:
                soil_pca = None
            
            # NLCD
            row, col = self.nlcd_src.index(x_5070, y_5070)
            if 0 <= row < self.nlcd_src.height and 0 <= col < self.nlcd_src.width:
                window = Window(col, row, 1, 1)
                nlcd_class = int(self.nlcd_src.read(1, window=window)[0, 0])
            else:
                nlcd_class = None
            
            # Ecoregion
            row, col = self.ecoregion_src.index(x_5070, y_5070)
            if 0 <= row < self.ecoregion_src.height and 0 <= col < self.ecoregion_src.width:
                window = Window(col, row, 1, 1)
                ecoregion_code = int(self.ecoregion_src.read(1, window=window)[0, 0])
            else:
                ecoregion_code = None
            
            return elevation, climate_pca, soil_pca, nlcd_class, ecoregion_code
            
        except Exception as e:
            logger.error(f"Error sampling rasters at ({x_5070:.2f}, {y_5070:.2f}): {e}")
            return None, None, None, None, None
    
    def close(self):
        """Close raster handles."""
        logger.info("Closing static rasters")
        self.elevation_src.close()
        self.climate_pca_src.close()
        self.soil_pca_src.close()
        self.nlcd_src.close()
        self.ecoregion_src.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to .pt file.
        config_path: Path to YAML config.
        device: Torch device.
        
    Returns:
        Eval mode model.
    """
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Creating model architecture")
    model = create_model(config)
    
    logger.info(f"Loading weights from {checkpoint_path}")
    
    # Try loading with weights_only=True first (secure, for clean weight files)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Loaded weights securely (weights_only=True)")
    except Exception as e:
        # Fall back to weights_only=False for legacy checkpoints
        logger.warning(f"Secure loading failed, using legacy mode: {e}")
        logger.warning("Consider converting checkpoint to weights-only format using scripts/convert_checkpoint_to_weights.py")
        
        # Add safe globals for numpy objects commonly found in checkpoints
        try:
            import numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except:
            pass
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.info("Loaded checkpoint with weights_only=False")
        except Exception as load_error:
            logger.error(f"Failed to load checkpoint: {load_error}")
            raise
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded weights-only file")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    return model


def extract_doy_from_filename(filename: str) -> int:
    """
    Get DOY from filename.
    
    Handles acquisition_processing and acquisition-only formats.
    
    Args:
        filename: NAIP filename.
        
    Returns:
        Day of year.
    """
    # Extract dates (YYYYMMDD)
    date_pattern = r'(\d{8})'
    matches = re.findall(date_pattern, filename)
    
    if not matches:
        raise ValueError(f"No date found in filename: {filename}")
    
    # Use first date (acquisition date)
    date_str = matches[0]
    
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        doy = date_obj.timetuple().tm_yday
        logger.debug(f"Extracted DOY {doy} from date {date_str} in {filename}")
        return doy
    except ValueError as e:
        raise ValueError(f"Invalid date format {date_str} in filename: {filename}") from e


def create_distance_weight_mask(
    chip_height: int,
    chip_width: int,
    clip_pixels: int
) -> np.ndarray:
    """
    Distance-based blend weights.
    
    Linear ramp from 0.0 (edge) to 1.0 (center).
    
    Args:
        chip_height: Chip height.
        chip_width: Chip width.
        clip_pixels: Edge clip amount.
        
    Returns:
        Weight mask.
    """
    # Start with all ones
    weights = np.ones((chip_height, chip_width), dtype=np.float32)
    
    if clip_pixels == 0:
        return weights
    
    # Create distance arrays from each edge
    row_indices = np.arange(chip_height)
    col_indices = np.arange(chip_width)
    
    # Distance from top edge (0 at top, increases downward)
    dist_from_top = row_indices.astype(np.float32)
    
    # Distance from bottom edge (0 at bottom, increases upward)
    dist_from_bottom = (chip_height - 1 - row_indices).astype(np.float32)
    
    # Distance from left edge (0 at left, increases rightward)
    dist_from_left = col_indices.astype(np.float32)
    
    # Distance from right edge (0 at right, increases leftward)
    dist_from_right = (chip_width - 1 - col_indices).astype(np.float32)
    
    # Create linear ramps in the clip zones
    # Weight = min(distance_from_edge, clip_pixels) / clip_pixels
    # This gives 0.0 at the edge and 1.0 at clip_pixels distance
    
    row_weight_top = np.minimum(dist_from_top, clip_pixels) / clip_pixels
    row_weight_bottom = np.minimum(dist_from_bottom, clip_pixels) / clip_pixels
    col_weight_left = np.minimum(dist_from_left, clip_pixels) / clip_pixels
    col_weight_right = np.minimum(dist_from_right, clip_pixels) / clip_pixels
    
    # Combine weights (minimum ensures smooth transition at corners)
    for i in range(chip_height):
        weights[i, :] *= row_weight_top[i]
        weights[i, :] *= row_weight_bottom[i]
    
    for j in range(chip_width):
        weights[:, j] *= col_weight_left[j]
        weights[:, j] *= col_weight_right[j]
    
    return weights


def create_chip_grid(
    width: int,
    height: int,
    chip_size: int,
    overlap_fraction: float
) -> List[Tuple[int, int, int, int]]:
    """
    Generate chip coordinates.
    
    Args:
        width: Image width.
        height: Image height.
        chip_size: Chip size.
        overlap_fraction: Overlap fraction.
        
    Returns:
        List of (row_start, col_start, row_end, col_end).
    """
    # Check chip size
    if chip_size % 16 != 0:
        raise ValueError(f"Chip size must be multiple of 16, got {chip_size}")
    
    overlap_pixels = int(chip_size * overlap_fraction)
    stride = chip_size - overlap_pixels
    
    chips = []
    
    # Generate chip positions
    row = 0
    while row < height:
        col = 0
        while col < width:
            # Calculate chip bounds
            row_end = min(row + chip_size, height)
            col_end = min(col + chip_size, width)
            
            # Adjust start if we're at the edge
            if row_end == height and row_end - row < chip_size:
                row_start = max(0, height - chip_size)
            else:
                row_start = row
                
            if col_end == width and col_end - col < chip_size:
                col_start = max(0, width - chip_size)
            else:
                col_start = col
            
            chips.append((row_start, col_start, row_end, col_end))
            
            # Move to next column
            if col_end == width:
                break
            col += stride
        
        # Move to next row
        if row_end == height:
            break
        row += stride
    
    logger.info(f"Created {len(chips)} chips of size {chip_size}x{chip_size} "
                f"with {overlap_fraction*100:.0f}% overlap (stride={stride})")
    
    return chips


def chip_bounds_to_epsg5070(
    chip_bounds: Tuple[float, float, float, float],
    src_crs: rasterio.crs.CRS
) -> Tuple[float, float]:
    """
    Convert bounds to center EPSG:5070.
    
    Args:
        chip_bounds: (left, bottom, right, top).
        src_crs: Source CRS.
        
    Returns:
        (x, y) center coordinates.
    """
    left, bottom, right, top = chip_bounds
    center_x = (left + right) / 2
    center_y = (bottom + top) / 2
    
    # Transform to EPSG:5070
    dst_crs = rasterio.crs.CRS.from_epsg(5070)
    
    if src_crs == dst_crs:
        return center_x, center_y
    
    # Transform bounds to get center point
    transformed = transform_bounds(src_crs, dst_crs, left, bottom, right, top)
    center_x_5070 = (transformed[0] + transformed[2]) / 2
    center_y_5070 = (transformed[1] + transformed[3]) / 2
    
    return center_x_5070, center_y_5070


def extract_auxiliary_features(
    x_5070: float,
    y_5070: float,
    doy: int,
    raster_handler: StaticRasterHandler
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Get aux features at point.
    
    Args:
        x_5070: X coordinate.
        y_5070: Y coordinate.
        doy: DOY.
        raster_handler: StaticRasterHandler.
        
    Returns:
        Feature dict or None.
    """
    try:
        # Sample rasters
        elevation, climate_pca, soil_pca, nlcd_class, ecoregion_code = raster_handler.sample_point(x_5070, y_5070)
        
        # Check if all features were successfully extracted
        if elevation is None or climate_pca is None or soil_pca is None:
            logger.warning(f"Missing continuous features at ({x_5070:.2f}, {y_5070:.2f})")
            return None
        
        if nlcd_class is None or ecoregion_code is None:
            logger.warning(f"Missing categorical features at ({x_5070:.2f}, {y_5070:.2f})")
            return None
        
        # Process continuous features
        features = []
        
        # Climate PCA (6 features) - normalized by std dev
        for i, pca_val in enumerate(climate_pca):
            normalized = pca_val / PCA_STD_DEVS[i]
            features.append(normalized)
        
        # Soil PCA (6 features) - normalized by std dev
        for i, soil_val in enumerate(soil_pca):
            normalized = soil_val / SOIL_PCA_STD_DEVS[i]
            features.append(normalized)
        
        # Elevation (normalized to 0-1)
        elevation_norm = np.clip(elevation / 4000.0, 0, 1)
        features.append(elevation_norm)
        
        # Day of year (circular encoding)
        features.append(np.sin(2 * np.pi * doy / 365))
        features.append(np.cos(2 * np.pi * doy / 365))
        
        continuous = torch.tensor(features, dtype=torch.float32)
        
        # Process categorical features
        if nlcd_class not in NLCD_TO_IDX:
            logger.warning(f"Unknown NLCD class {nlcd_class} at ({x_5070:.2f}, {y_5070:.2f})")
            return None
        
        nlcd_idx = torch.tensor(NLCD_TO_IDX[nlcd_class], dtype=torch.long)
        ecoregion_idx = torch.tensor(int(ecoregion_code), dtype=torch.long)
        
        return {
            'continuous': continuous,
            'nlcd_idx': nlcd_idx,
            'ecoregion_idx': ecoregion_idx
        }
        
    except Exception as e:
        logger.error(f"Error extracting features at ({x_5070:.2f}, {y_5070:.2f}): {e}")
        return None


def process_naip_quad(
    naip_quad_path: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    raster_handler: StaticRasterHandler,
    chip_size: int = 432,
    chip_overlap: float = 0.2,
    edge_clip_meters: float = 0.0,
    dry_run: bool = False
) -> Dict[str, any]:
    """
    Process NAIP quad to CHM.
    
    Args:
        naip_quad_path: Input path.
        output_dir: Output directory.
        model: Loaded model.
        device: Inference device.
        raster_handler: Raster handler.
        chip_size: Chip size.
        chip_overlap: Overlap fraction.
        edge_clip_meters: Edge clip distance.
        dry_run: Validate only.
        
    Returns:
        Stats dict.
    """
    import time
    start_time = time.time()
    
    logger.info(f"Processing NAIP quad: {naip_quad_path.name}")
    
    # Extract DOY from filename
    try:
        doy = extract_doy_from_filename(naip_quad_path.name)
        logger.info(f"Extracted DOY: {doy}")
    except ValueError as e:
        logger.error(f"Failed to extract DOY: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Open NAIP quad
    try:
        with rasterio.open(naip_quad_path) as src:
            # Read metadata
            width = src.width
            height = src.height
            crs = src.crs
            transform = src.transform
            bounds = src.bounds
            
            logger.info(f"NAIP dimensions: {width}x{height}")
            logger.info(f"CRS: {crs}")
            logger.info(f"Bounds: {bounds}")
            
            if dry_run:
                logger.info("DRY RUN: Skipping actual processing")
                return {
                    'status': 'dry_run',
                    'width': width,
                    'height': height,
                    'crs': str(crs),
                    'doy': doy
                }
            
            # Check resolution and calculate target dimensions (0.6m)
            src_res_x = abs(src.transform.a)
            src_res_y = abs(src.transform.e)
            target_res = 0.6
            
            # If resolution differs by more than 1% from 0.6m, resample
            if abs(src_res_x - target_res) > 0.006 or abs(src_res_y - target_res) > 0.006:
                logger.info(f"Resampling input from {src_res_x:.2f}m to {target_res}m")
                
                # Calculate new dimensions
                new_width = int(width * (src_res_x / target_res))
                new_height = int(height * (src_res_y / target_res))
                
                # Calculate new transform
                from rasterio.transform import Affine
                new_transform = src.transform * Affine.scale(
                    (width / new_width),
                    (height / new_height)
                )
                
                # Update metadata variables
                width = new_width
                height = new_height
                transform = new_transform
                
                # Read and resample
                # Handle band counts
                num_bands = src.count
                
                if num_bands == 4:
                    logger.info("Input has 4 bands (RGBN), assuming full validity")
                    # Read RGBN with bilinear
                    rgbn_data = src.read(
                        out_shape=(4, height, width),
                        resampling=Resampling.bilinear
                    )
                    # Create full mask
                    mask = np.ones((height, width), dtype=bool)
                    
                elif num_bands >= 5:
                    logger.info(f"Input has {num_bands} bands, using first 4 as RGBN and 5th as mask")
                    # Read RGBN with bilinear
                    rgbn_data = src.read(
                        indexes=[1, 2, 3, 4],
                        out_shape=(4, height, width),
                        resampling=Resampling.bilinear
                    )
                    # Read mask with nearest neighbor
                    mask_data = src.read(
                        indexes=5,
                        out_shape=(1, height, width),
                        resampling=Resampling.nearest
                    )
                    mask = mask_data[0] > 0
                    
                else:
                    raise ValueError(f"Unsupported band count: {num_bands}. Expected 4 or >=5.")
                
            else:
                logger.info(f"Input resolution {src_res_x:.2f}m is close to target 0.6m, no resampling needed")
                
                # Read original data
                num_bands = src.count
                if num_bands == 4:
                    logger.info("Input has 4 bands (RGBN), assuming full validity")
                    rgbn_data = src.read()
                    mask = np.ones((height, width), dtype=bool)
                elif num_bands >= 5:
                    logger.info(f"Input has {num_bands} bands, using first 4 as RGBN and 5th as mask")
                    data = src.read()
                    rgbn_data = data[:4]
                    mask = data[4] > 0
                else:
                    raise ValueError(f"Unsupported band count: {num_bands}. Expected 4 or >=5.")

            # Normalize RGBN to 0-1
            rgbn = rgbn_data.astype(np.float32) / 255.0
            
    except Exception as e:
        logger.error(f"Error reading NAIP quad: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Create chip grid
    try:
        chips = create_chip_grid(width, height, chip_size, chip_overlap)
    except ValueError as e:
        logger.error(f"Error creating chip grid: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Calculate per-chip edge clipping amount (1/4 of overlap to leave room for blending)
    overlap_pixels = int(chip_size * chip_overlap)
    clip_pixels = overlap_pixels // 4
    blend_pixels = overlap_pixels - (2 * clip_pixels)
    logger.info(f"Tile blending strategy: clip {clip_pixels}px from each edge, "
                f"blend over {blend_pixels}px in overlap regions (overlap={overlap_pixels}px)")
    
    # Pre-compute distance weight mask for blending
    base_weight_mask = create_distance_weight_mask(chip_size, chip_size, clip_pixels)
    
    # Initialize output arrays
    chm_output = np.zeros((height, width), dtype=np.float32)
    weight_output = np.zeros((height, width), dtype=np.float32)
    
    # Process each chip
    processed_chips = 0
    failed_chips = 0
    
    for chip_idx, (row_start, col_start, row_end, col_end) in enumerate(chips):
        try:
            # Extract chip from NAIP
            chip_data = rgbn[:, row_start:row_end, col_start:col_end]
            chip_height, chip_width = chip_data.shape[1], chip_data.shape[2]
            
            # Get chip bounds in native CRS
            chip_window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
            chip_bounds = rasterio.windows.bounds(chip_window, transform)
            
            # Convert chip center to EPSG:5070 for metadata sampling
            x_5070, y_5070 = chip_bounds_to_epsg5070(chip_bounds, crs)
            
            # Extract auxiliary features using persistent raster handles
            aux_features = extract_auxiliary_features(
                x_5070, y_5070, doy, raster_handler
            )
            
            if aux_features is None:
                logger.warning(f"Skipping chip {chip_idx+1}/{len(chips)} - missing features")
                failed_chips += 1
                continue
            
            # Prepare model inputs
            image_tensor = torch.from_numpy(chip_data).unsqueeze(0).to(device)
            continuous = aux_features['continuous'].unsqueeze(0).to(device)
            nlcd_idx = aux_features['nlcd_idx'].unsqueeze(0).to(device)
            ecoregion_idx = aux_features['ecoregion_idx'].unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                prediction = model(image_tensor, continuous, nlcd_idx, ecoregion_idx)
            
            # Convert to numpy
            chm_chip = prediction.squeeze().cpu().numpy()
            
            # Get weight mask for this chip
            if chip_height == chip_size and chip_width == chip_size:
                chip_weight_mask = base_weight_mask.copy()
            else:
                chip_weight_mask = create_distance_weight_mask(chip_height, chip_width, clip_pixels)
            
            # Apply distance-weighted blending
            weighted_prediction = chm_chip * chip_weight_mask
            
            # Add weighted prediction to output
            chm_output[row_start:row_end, col_start:col_end] += weighted_prediction
            weight_output[row_start:row_end, col_start:col_end] += chip_weight_mask
            
            processed_chips += 1
            
            if (chip_idx + 1) % 10 == 0:
                logger.info(f"Processed {chip_idx+1}/{len(chips)} chips")
                
        except Exception as e:
            logger.error(f"Error processing chip {chip_idx+1}: {e}")
            failed_chips += 1
            continue
    
    logger.info(f"Processed {processed_chips}/{len(chips)} chips successfully, {failed_chips} failed")
    
    # Average overlapping regions
    valid_weights = weight_output > 0
    chm_output[valid_weights] /= weight_output[valid_weights]
    
    # Apply original mask
    chm_output[~mask] = 0
    
    # Post-process: convert to centimeters and clip
    chm_output = chm_output * 100.0
    chm_output = np.clip(chm_output, 0, 12000)
    
    # Convert to UInt16
    chm_uint16 = chm_output.astype(np.uint16)
    
    # Set NoData where mask is invalid
    chm_uint16[~mask] = 65535
    
    # Apply edge clipping if specified
    if edge_clip_meters > 0:
        pixel_size = abs(transform.a)
        clip_pixels = int(edge_clip_meters / pixel_size)
        
        if clip_pixels * 2 >= min(height, width):
            logger.warning(f"Edge clip ({edge_clip_meters}m = {clip_pixels}px) too large for image "
                          f"({width}x{height}), skipping clipping")
            clip_pixels = 0
        
        if clip_pixels > 0:
            chm_clipped = chm_uint16[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels]
            clipped_height, clipped_width = chm_clipped.shape
            
            from rasterio.transform import Affine
            clipped_transform = transform * Affine.translation(clip_pixels, clip_pixels)
            
            logger.info(f"Applied edge clipping: {edge_clip_meters}m ({clip_pixels}px) from all edges")
            logger.info(f"Original dimensions: {width}x{height}, Clipped: {clipped_width}x{clipped_height}")
            
            chm_uint16 = chm_clipped
            width = clipped_width
            height = clipped_height
            transform = clipped_transform
    
    # Save as Cloud-Optimized GeoTIFF
    output_filename = output_dir / f"{naip_quad_path.stem}_chm.tif"
    temp_filename = output_dir / f"{naip_quad_path.stem}_chm_temp.tif"
    
    try:
        # Write regular GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'nodata': 65535,
            'compress': 'DEFLATE',
            'predictor': 2,
            'BIGTIFF': 'IF_SAFER'
        }
        
        with rasterio.open(temp_filename, 'w', **profile) as dst:
            dst.write(chm_uint16, 1)
        
        logger.info(f"Wrote temporary GeoTIFF to {temp_filename}")
        
        # Convert to COG
        import subprocess
        
        min_dimension = min(width, height)
        overview_level = 2
        while min_dimension // (2 ** overview_level) >= 2:
            overview_level += 1
        
        logger.info(f"Converting to COG with {overview_level} overview levels")
        
        cmd = [
            'rio', 'cogeo', 'create',
            str(temp_filename),
            str(output_filename),
            '--overview-level', str(overview_level),
            '--blocksize', '512',
            '--overview-resampling', 'average'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully created COG: {output_filename}")
            temp_filename.unlink()
            logger.debug(f"Deleted temporary file: {temp_filename}")
        else:
            logger.error(f"COG creation failed: {result.stderr}")
            raise RuntimeError(f"COG creation failed: {result.stderr}")
        
        elapsed_seconds = time.time() - start_time
        
        return {
            'status': 'success',
            'output_path': str(output_filename),
            'processed_chips': processed_chips,
            'failed_chips': failed_chips,
            'total_chips': len(chips),
            'elapsed_seconds': elapsed_seconds
        }
        
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        return {'status': 'failed', 'error': str(e)}
