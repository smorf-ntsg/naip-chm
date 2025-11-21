"""
Streaming inference for NAIP CHM using Google Earth Engine.
Fetches data on-the-fly and runs inference without downloading large input rasters.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import ee
import numpy as np
import rasterio
import torch
import yaml
import pyproj
from rasterio.transform import from_bounds, Affine, array_bounds
from rasterio.crs import CRS

from src.model import create_model
from src.dataset import NLCD_TO_IDX, PCA_STD_DEVS, SOIL_PCA_STD_DEVS
from src.inference_utils import create_chip_grid, create_distance_weight_mask
from src.utils.gee import get_naip_doqq_geometry, fetch_chip_data, fetch_conditioning_vector

logger = logging.getLogger(__name__)

# Suppress noisy rasterio warnings about 4-band TIFFs
logging.getLogger("rasterio").setLevel(logging.ERROR)

class GEEInferenceStreamer:
    """
    Orchestrates streaming inference from GEE.
    """
    
    def __init__(self, model_path: Path, config_path: Path, device: torch.device):
        """
        Initialize streamer with model.
        """
        self.device = device
        self.model = self._load_model(model_path, config_path)
        
    def _load_model(self, checkpoint_path: Path, config_path: Path) -> torch.nn.Module:
        """Load model from checkpoint."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        model = create_model(config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        return model

    def stream_inference(
        self,
        lat: float,
        lon: float,
        year: int,
        output_dir: Path,
        project_id: str,
        chip_size: int = 432,
        overlap_fraction: float = 0.2
    ) -> Dict[str, Any]:
        """
        Run streaming inference for the DOQQ covering (lat, lon).
        """
        from src.utils.gee import initialize_gee
        initialize_gee(project_id)
        
        # 1. Get DOQQ Geometry and Metadata
        image_id, geometry, info = get_naip_doqq_geometry(lat, lon, year)
        logger.info(f"Processing DOQQ: {image_id}")
        
        # Get Native Projection Info
        # NAIP usually has 4 bands. We use the first band for metadata.
        band0 = info['bands'][0]
        native_crs_code = band0['crs']
        native_transform_list = band0['crs_transform'] 
        # GEE transform: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
        native_transform = Affine(*native_transform_list)
        native_width = band0['dimensions'][0]
        native_height = band0['dimensions'][1]
        
        logger.info(f"Native CRS: {native_crs_code}")
        logger.info(f"Native Dimensions: {native_width}x{native_height}")
        logger.info(f"Native Transform: {native_transform}")

        # 2. Determine Target Grid (0.6m resolution)
        # Calculate native resolution
        res_x = abs(native_transform.a)
        res_y = abs(native_transform.e)
        target_res = 0.6
        
        # Check if resampling is needed
        if abs(res_x - target_res) > 0.006 or abs(res_y - target_res) > 0.006:
            logger.info(f"Resampling from {res_x:.2f}m to {target_res}m")
            
            # Calculate native bounds
            left, bottom, right, top = array_bounds(native_height, native_width, native_transform)
            
            # Calculate new dimensions based on target resolution
            width = int((right - left) / target_res)
            height = int((top - bottom) / target_res)
            
            # Construct new transform
            transform = from_bounds(left, bottom, right, top, width, height)
        else:
            logger.info(f"Native resolution {res_x:.2f}m matches target. Using native grid.")
            width = native_width
            height = native_height
            transform = native_transform
            
        crs = CRS.from_string(native_crs_code)
        logger.info(f"Target Grid: {width}x{height} pixels")
        
        # Setup coordinate transformer for aux data (Native -> Lat/Lon)
        # We need this to query conditioning data which requires Lat/Lon (or we pass point object)
        # Since fetch_conditioning_vector creates a Point from lat/lon, we transform to lat/lon.
        to_latlon = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        # 3. Generate Chip Grid
        chips = create_chip_grid(width, height, chip_size, overlap_fraction)
        
        # 4. Prepare Blending
        overlap_pixels = int(chip_size * overlap_fraction)
        clip_pixels = overlap_pixels // 4
        base_weight_mask = create_distance_weight_mask(chip_size, chip_size, clip_pixels)
        
        # Output buffers
        chm_output = np.zeros((height, width), dtype=np.float32)
        weight_output = np.zeros((height, width), dtype=np.float32)
        
        # 5. Streaming Loop
        # Helper to fetch data for a single chip
        def fetch_chip_task(chip_args):
            c_idx, r_start, c_start, r_end, c_end = chip_args
            
            # Calculate chip bounds in Target CRS (which is Native CRS aligned)
            # Using rasterio.windows.bounds equivalent logic
            # Affine transform: col -> x, row -> y
            # Top-Left
            left, top = transform * (c_start, r_start)
            # Bottom-Right
            right, bottom = transform * (c_end, r_end)
            # Note: 'top' is usually > 'bottom' in projected coords (y increases North)
            # But Affine transform handles the sign of 'e' (usually negative).
            
            # Create EE geometry for the chip in Native CRS
            # ee.Geometry.Rectangle takes [xMin, yMin, xMax, yMax]
            x_min, x_max = min(left, right), max(left, right)
            y_min, y_max = min(bottom, top), max(bottom, top)
            
            chip_geom = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], native_crs_code, False)
            
            # Get Center for conditioning
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Transform to Lat/Lon
            lon_c, lat_c = to_latlon.transform(center_x, center_y)
            
            try:
                # Fetch raster
                # We request the specific shape we expect
                req_height = r_end - r_start
                req_width = c_end - c_start
                
                img_data = fetch_chip_data(
                    image_id, 
                    chip_geom, 
                    shape=(req_height, req_width),
                    crs=native_crs_code
                )
                
                # Fetch vector
                img_date = ee.Image(image_id).get('system:time_start').getInfo()
                doy = time.gmtime(img_date / 1000).tm_yday
                
                cond_data = fetch_conditioning_vector(lat_c, lon_c, doy)
                
                return {
                    'idx': c_idx,
                    'img': img_data,
                    'cond': cond_data,
                    'coords': (r_start, c_start, r_end, c_end)
                }
            except Exception as e:
                logger.error(f"Failed to fetch chip {c_idx}: {e}")
                return None

        # Use ThreadPoolExecutor
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            futures = []
            for i, (r_s, c_s, r_e, c_e) in enumerate(chips):
                futures.append(executor.submit(fetch_chip_task, (i, r_s, c_s, r_e, c_e)))
            
            from concurrent.futures import as_completed
            from tqdm import tqdm
            
            for future in tqdm(as_completed(futures), total=len(chips), desc="Inference"):
                try:
                    result = future.result()
                    if result is None:
                        failed_count += 1
                        continue
                    
                    if result['cond'] is None:
                        failed_count += 1
                        logger.warning(f"Chip {result['idx']} skipped: Missing conditioning data")
                        continue
                    
                    # Unpack
                    img_data = result['img']
                    cond_data = result['cond']
                    r_s, c_s, r_e, c_e = result['coords']
                    
                    # Prepare inputs
                    image_tensor = torch.from_numpy(img_data).unsqueeze(0).to(self.device)
                    
                    # Conditioning
                    feats = self._process_features(cond_data)
                    if feats is None: 
                        failed_count += 1
                        logger.warning(f"Chip {result['idx']} skipped: Feature processing failed (NLCD/Eco check)")
                        continue
                    
                    continuous, nlcd, eco = feats
                    continuous = continuous.unsqueeze(0).to(self.device)
                    nlcd = nlcd.unsqueeze(0).to(self.device)
                    eco = eco.unsqueeze(0).to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        pred = self.model(image_tensor, continuous, nlcd, eco)
                    
                    chm_chip = pred.squeeze().cpu().numpy()
                    
                    # Blending
                    h_chip, w_chip = chm_chip.shape
                    if h_chip == chip_size and w_chip == chip_size:
                        mask = base_weight_mask
                    else:
                        mask = create_distance_weight_mask(h_chip, w_chip, clip_pixels)
                    
                    chm_output[r_s:r_e, c_s:c_e] += chm_chip * mask
                    weight_output[r_s:r_e, c_s:c_e] += mask
                    
                    processed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Processing failed for chip: {e}")

        logger.info(f"Processing complete. Processed: {processed_count}, Failed: {failed_count} ({failed_count/len(chips)*100:.1f}%)")
        
        # Finalize
        valid_weights = weight_output > 0
        chm_output[valid_weights] /= weight_output[valid_weights]
        
        # 6. Post-processing
        # Convert to cm
        chm_output = chm_output * 100.0
        chm_output = np.clip(chm_output, 0, 65535) # Safety clip for uint16
        chm_uint16 = chm_output.astype(np.uint16)
        
        # Edge Trimming (30m)
        # Local inference clips 30m from edges
        trim_pixels = int(30.0 / target_res) # 30 / 0.6 = 50 pixels
        if trim_pixels > 0 and trim_pixels * 2 < min(width, height):
            logger.info(f"Trimming {trim_pixels} pixels ({30}m) from edges")
            chm_uint16 = chm_uint16[trim_pixels:-trim_pixels, trim_pixels:-trim_pixels]
            # Update transform
            transform = transform * Affine.translation(trim_pixels, trim_pixels)
            height, width = chm_uint16.shape
        
        # 7. Save as COG
        output_path = output_dir / f"{image_id.split('/')[-1]}_chm.tif"
        self._save_cog(chm_uint16, output_path, transform, crs)
        
        return {'status': 'success', 'path': str(output_path), 'processed': processed_count}

    def _process_features(self, row: Dict) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Normalize features."""
        try:
            feats = []
            # Climate
            for i in range(1, 7):
                val = row.get(f'climate_pca_{i}')
                feats.append(val / PCA_STD_DEVS[i-1])
            # Soil
            for i in range(1, 7):
                val = row.get(f'soil_pca_{i}')
                feats.append(val / SOIL_PCA_STD_DEVS[i-1])
            # Elevation
            elev = row.get('elevation')
            feats.append(np.clip(elev / 4000.0, 0, 1))
            # DOY
            doy = row.get('doy')
            feats.append(np.sin(2 * np.pi * doy / 365))
            feats.append(np.cos(2 * np.pi * doy / 365))
            
            continuous = torch.tensor(feats, dtype=torch.float32)
            
            nlcd_val = int(row.get('nlcd'))
            if nlcd_val not in NLCD_TO_IDX: return None
            nlcd = torch.tensor(NLCD_TO_IDX[nlcd_val], dtype=torch.long)
            
            eco_val = int(row.get('ecoregion'))
            eco = torch.tensor(eco_val, dtype=torch.long)
            
            return continuous, nlcd, eco
        except Exception:
            return None

    def _save_cog(self, data, path, transform, crs):
        """Save array as COG."""
        import subprocess
        temp_path = path.with_suffix('.temp.tif')
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'width': data.shape[1],
            'height': data.shape[0],
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'DEFLATE',
            'predictor': 2,
            'nodata': 65535
        }
        
        with rasterio.open(temp_path, 'w', **profile) as dst:
            dst.write(data, 1)
            
        # Convert to COG
        cmd = [
            'rio', 'cogeo', 'create',
            str(temp_path),
            str(path),
            '--overview-level', '5'
        ]
        subprocess.run(cmd, check=True)
        temp_path.unlink()
        logger.info(f"Saved COG to {path}")
