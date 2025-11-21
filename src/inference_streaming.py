"""
Streaming inference for NAIP CHM using Google Earth Engine.
Fetches data on-the-fly and runs inference without downloading large input rasters.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ee
import numpy as np
import rasterio
import torch
import yaml
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from src.model import create_model
from src.dataset import NLCD_TO_IDX, PCA_STD_DEVS, SOIL_PCA_STD_DEVS
from src.inference_utils import create_chip_grid, create_distance_weight_mask
from src.utils.gee import get_naip_doqq_geometry, fetch_chip_data, fetch_conditioning_vector

logger = logging.getLogger(__name__)

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
        
        # 1. Get DOQQ Geometry
        image_id, geometry, info = get_naip_doqq_geometry(lat, lon, year)
        logger.info(f"Processing DOQQ: {image_id}")
        
        # Get bounds in EPSG:5070
        # We transform the geometry to 5070 to get accurate meter dimensions
        geom_5070 = geometry.transform('EPSG:5070', 1) # 1m error margin
        coords = geom_5070.bounds().getInfo()['coordinates'][0]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate dimensions at 0.6m resolution
        pixel_size = 0.6
        width = int((max_x - min_x) / pixel_size)
        height = int((max_y - min_y) / pixel_size)
        
        logger.info(f"DOQQ Grid: {width}x{height} pixels ({pixel_size}m res)")
        
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        crs = CRS.from_epsg(5070)
        
        # 2. Generate Chip Grid
        chips = create_chip_grid(width, height, chip_size, overlap_fraction)
        
        # 3. Prepare Blending
        overlap_pixels = int(chip_size * overlap_fraction)
        clip_pixels = overlap_pixels // 4
        base_weight_mask = create_distance_weight_mask(chip_size, chip_size, clip_pixels)
        
        # Output buffers
        chm_output = np.zeros((height, width), dtype=np.float32)
        weight_output = np.zeros((height, width), dtype=np.float32)
        
        # 4. Streaming Loop
        # Helper to fetch data for a single chip
        def fetch_chip_task(chip_args):
            c_idx, r_start, c_start, r_end, c_end = chip_args
            
            # Calculate chip bounds in EPSG:5070
            left = min_x + c_start * pixel_size
            top = max_y - r_start * pixel_size
            right = min_x + c_end * pixel_size
            bottom = max_y - r_end * pixel_size
            
            # Create EE geometry for the chip
            # Note: 'top' is ymax, 'bottom' is ymin. But in raster coords, row 0 is top.
            chip_geom = ee.Geometry.Rectangle([left, bottom, right, top], 'EPSG:5070', False)
            
            # Get Center for conditioning
            center_x = (left + right) / 2
            center_y = (bottom + top) / 2
            # Convert center to lat/lon for conditioning vector fetch (helper expects lat/lon if needed?)
            # Actually fetch_conditioning_vector in gee.py expects lat/lon.
            # So we need to transform back to 4326.
            # Or update fetch_conditioning_vector to take 5070 coords. 
            # The current implementation expects lat/lon. Let's transform.
            import pyproj
            transformer = pyproj.Transformer.from_crs(5070, 4326, always_xy=True)
            lon_c, lat_c = transformer.transform(center_x, center_y)
            
            try:
                # Fetch raster
                img_data = fetch_chip_data(
                    image_id, 
                    chip_geom, 
                    shape=(r_end-r_start, c_end-c_start)
                )
                
                # Fetch vector
                # Calculate DOY from image ID if possible or assume mid-summer?
                # NAIP usually has date in metadata. 'system:time_start'.
                img_date = ee.Image(image_id).get('system:time_start').getInfo()
                # Convert millis to DOY
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
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks (or batch them if memory is tight, but 8 threads is fine)
            futures = []
            for i, (r_s, c_s, r_e, c_e) in enumerate(chips):
                futures.append(executor.submit(fetch_chip_task, (i, r_s, c_s, r_e, c_e)))
            
            from concurrent.futures import as_completed
            from tqdm import tqdm
            
            for future in tqdm(as_completed(futures), total=len(chips), desc="Inference"):
                result = future.result()
                if result is None or result['cond'] is None:
                    continue
                
                # Unpack
                img_data = result['img']
                cond_data = result['cond']
                r_s, c_s, r_e, c_e = result['coords']
                
                # Prepare inputs
                # Image
                image_tensor = torch.from_numpy(img_data).unsqueeze(0).to(self.device)
                
                # Conditioning
                try:
                    feats = self._process_features(cond_data)
                    if feats is None: continue
                    
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
                    logger.error(f"Inference failed for chip {result['idx']}: {e}")
        
        # Finalize
        valid_weights = weight_output > 0
        chm_output[valid_weights] /= weight_output[valid_weights]
        
        # 5. Post-processing
        # Convert to cm
        chm_output = chm_output * 100.0
        chm_output = np.clip(chm_output, 0, 65535) # Safety clip for uint16
        chm_uint16 = chm_output.astype(np.uint16)
        
        # Edge Trimming (30m)
        trim_pixels = int(30.0 / pixel_size) # 30 / 0.6 = 50 pixels
        if trim_pixels > 0 and trim_pixels * 2 < min(width, height):
            logger.info(f"Trimming {trim_pixels} pixels ({30}m) from edges")
            chm_uint16 = chm_uint16[trim_pixels:-trim_pixels, trim_pixels:-trim_pixels]
            # Update transform
            from rasterio.transform import Affine
            transform = transform * Affine.translation(trim_pixels, trim_pixels)
            height, width = chm_uint16.shape
        
        # 6. Save as COG
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
        # Ideally use rio-cogeo python API instead of subprocess if possible, but subprocess is robust
        # Or simpler:
        cmd = [
            'rio', 'cogeo', 'create',
            str(temp_path),
            str(path),
            '--overview-level', '5'
        ]
        subprocess.run(cmd, check=True)
        temp_path.unlink()
        logger.info(f"Saved COG to {path}")
