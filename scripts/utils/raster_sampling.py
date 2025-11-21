"""
Raster sampling utilities.
Efficient point extraction from COGs.
"""

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from typing import Tuple, List, Optional, Union
import logging
from pathlib import Path


class RasterSampler:
    """Raster point sampling."""
    
    def __init__(self, elevation_path: str, pca_path: str, soil_pca_path: str):
        """
        Initialize raster paths.
        
        Args:
            elevation_path: Path to elevation COG.
            pca_path: Path to climate PCA COG.
            soil_pca_path: Path to soil PCA COG.
        """
        self.elevation_path = Path(elevation_path)
        self.pca_path = Path(pca_path)
        self.soil_pca_path = Path(soil_pca_path)
        self.logger = logging.getLogger(__name__)
        
        # Verify files exist
        if not self.elevation_path.exists():
            raise FileNotFoundError(f"Elevation raster not found: {elevation_path}")
        if not self.pca_path.exists():
            raise FileNotFoundError(f"PCA raster not found: {pca_path}")
        if not self.soil_pca_path.exists():
            raise FileNotFoundError(f"Soil PCA raster not found: {soil_pca_path}")
        
        # Store raster info
        self._elevation_src = None
        self._pca_src = None
        self._soil_pca_src = None
        self._elevation_info = None
        self._pca_info = None
        self._soil_pca_info = None
        
        self.logger.info(f"Initialized raster sampler with:")
        self.logger.info(f"  Elevation: {self.elevation_path}")
        self.logger.info(f"  PCA: {self.pca_path}")
        self.logger.info(f"  Soil PCA: {self.soil_pca_path}")
    
    def __enter__(self):
        """Open rasters."""
        try:
            self._elevation_src = rasterio.open(self.elevation_path)
            self._pca_src = rasterio.open(self.pca_path)
            self._soil_pca_src = rasterio.open(self.soil_pca_path)
            
            # Validate raster properties
            self._validate_rasters()
            
            return self
            
        except Exception as e:
            self._cleanup()
            raise e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close rasters."""
        self._cleanup()
    
    def _cleanup(self):
        """Close raster sources."""
        if self._elevation_src:
            self._elevation_src.close()
            self._elevation_src = None
        if self._pca_src:
            self._pca_src.close()
            self._pca_src = None
        if self._soil_pca_src:
            self._soil_pca_src.close()
            self._soil_pca_src = None
    
    def _validate_rasters(self):
        """Validate raster properties."""
        # Check elevation raster
        if self._elevation_src.count != 1:
            raise ValueError(f"Elevation raster should have 1 band, got {self._elevation_src.count}")
        
        # Check PCA raster
        if self._pca_src.count != 6:
            raise ValueError(f"PCA raster should have 6 bands, got {self._pca_src.count}")

        # Check Soil PCA raster
        if self._soil_pca_src.count != 6:
            raise ValueError(f"Soil PCA raster should have 6 bands, got {self._soil_pca_src.count}")
        
        # Check CRS
        if self._elevation_src.crs.to_string() != "EPSG:5070":
            self.logger.warning(f"Elevation CRS is {self._elevation_src.crs}, expected EPSG:5070")
        
        if self._pca_src.crs.to_string() != "EPSG:5070":
            self.logger.warning(f"PCA CRS is {self._pca_src.crs}, expected EPSG:5070")

        if self._soil_pca_src.crs.to_string() != "EPSG:5070":
            self.logger.warning(f"Soil PCA CRS is {self._soil_pca_src.crs}, expected EPSG:5070")
        
        self.logger.info("Raster validation completed successfully")
    
    def sample_point(self, x_5070: float, y_5070: float) -> Tuple[Optional[float], Optional[List[float]], Optional[List[float]]]:
        """
        Sample features at point.
        
        Args:
            x_5070: X coordinate in EPSG:5070.
            y_5070: Y coordinate in EPSG:5070.
            
        Returns:
            Tuple of (elevation, climate_pca, soil_pca) or (None, None, None).
        """
        try:
            # Sample elevation
            elevation = self._sample_single_band(self._elevation_src, x_5070, y_5070)
            
            # Sample climate PCA bands
            pca_values = self._sample_multi_band(self._pca_src, x_5070, y_5070)

            # Sample soil PCA bands
            soil_pca_values = self._sample_multi_band(self._soil_pca_src, x_5070, y_5070)
            
            return elevation, pca_values, soil_pca_values
            
        except Exception as e:
            self.logger.warning(f"Failed to sample point ({x_5070:.2f}, {y_5070:.2f}): {e}")
            return None, None, None
    
    def _sample_single_band(self, src: rasterio.DatasetReader, 
                           x: float, y: float) -> Optional[float]:
        """Sample single-band raster."""
        try:
            # Convert coordinates to pixel indices
            row, col = src.index(x, y)
            
            # Check bounds
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            
            # Read value
            window = rasterio.windows.Window(col, row, 1, 1)
            data = src.read(1, window=window)
            
            value = data[0, 0]
            
            # Handle NoData
            if src.nodata is not None and value == src.nodata:
                return None
            
            return float(value)
            
        except Exception:
            return None
    
    def _sample_multi_band(self, src: rasterio.DatasetReader, 
                          x: float, y: float) -> Optional[List[float]]:
        """Sample multi-band raster."""
        try:
            # Convert coordinates to pixel indices
            row, col = src.index(x, y)
            
            # Check bounds
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            
            # Read all bands at once
            window = rasterio.windows.Window(col, row, 1, 1)
            data = src.read(window=window)  # Shape: (bands, 1, 1)
            
            values = data[:, 0, 0]
            
            # Handle NoData
            if src.nodata is not None:
                values = np.where(values == src.nodata, np.nan, values)
                if np.any(np.isnan(values)):
                    return None
            
            return values.astype(float).tolist()
            
        except Exception:
            return None
    
    def sample_batch(self, coords_5070: np.ndarray, 
                    batch_size: int = 10000) -> Tuple[List[Optional[float]], List[Optional[List[float]]], List[Optional[List[float]]]]:
        """
        Batch sample features.
        
        Args:
            coords_5070: Array of shape (n, 2) with EPSG:5070 coordinates.
            batch_size: Batch size.
            
        Returns:
            Tuple of (elevations, pca_values_list, soil_pca_values_list).
        """
        n_points = len(coords_5070)
        elevations = []
        pca_values_list = []
        soil_pca_values_list = []
        
        self.logger.info(f"Sampling {n_points} points in batches of {batch_size}")
        
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            batch_coords = coords_5070[i:batch_end]
            
            batch_elevations = []
            batch_pca = []
            batch_soil_pca = []
            
            for x_5070, y_5070 in batch_coords:
                elevation, pca_values, soil_pca_values = self.sample_point(x_5070, y_5070)
                batch_elevations.append(elevation)
                batch_pca.append(pca_values)
                batch_soil_pca.append(soil_pca_values)
            
            elevations.extend(batch_elevations)
            pca_values_list.extend(batch_pca)
            soil_pca_values_list.extend(batch_soil_pca)
            
            if (i // batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed {batch_end}/{n_points} points")
        
        return elevations, pca_values_list, soil_pca_values_list
    
    def get_raster_info(self) -> dict:
        """Get raster metadata."""
        if not self._elevation_src or not self._pca_src or not self._soil_pca_src:
            raise RuntimeError("Raster sampler not initialized. Use with context manager.")
        
        return {
            'elevation': {
                'path': str(self.elevation_path),
                'crs': self._elevation_src.crs.to_string(),
                'bounds': self._elevation_src.bounds,
                'shape': (self._elevation_src.height, self._elevation_src.width),
                'resolution': self._elevation_src.res,
                'nodata': self._elevation_src.nodata
            },
            'pca': {
                'path': str(self.pca_path),
                'crs': self._pca_src.crs.to_string(),
                'bounds': self._pca_src.bounds,
                'shape': (self._pca_src.height, self._pca_src.width),
                'bands': self._pca_src.count,
                'resolution': self._pca_src.res,
                'nodata': self._pca_src.nodata
            },
            'soil_pca': {
                'path': str(self.soil_pca_path),
                'crs': self._soil_pca_src.crs.to_string(),
                'bounds': self._soil_pca_src.bounds,
                'shape': (self._soil_pca_src.height, self._soil_pca_src.width),
                'bands': self._soil_pca_src.count,
                'resolution': self._soil_pca_src.res,
                'nodata': self._soil_pca_src.nodata
            }
        }


def extract_features_at_points(elevation_path: str, pca_path: str, soil_pca_path: str,
                              coords_5070: np.ndarray) -> Tuple[List[Optional[float]], List[Optional[List[float]]], List[Optional[List[float]]]]:
    """
    Extract features at coordinates.
    
    Args:
        elevation_path: Path to elevation COG.
        pca_path: Path to PCA COG.
        soil_pca_path: Path to soil PCA COG.
        coords_5070: Array of EPSG:5070 coordinates.
        
    Returns:
        Tuple of (elevations, pca_values_list, soil_pca_values_list).
    """
    with RasterSampler(elevation_path, pca_path, soil_pca_path) as sampler:
        return sampler.sample_batch(coords_5070)


if __name__ == "__main__":
    # Test raster sampling
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock test with sample coordinates
    test_coords = np.array([
        [1000000, 2000000],  # Example EPSG:5070 coordinates
        [1100000, 2100000],
        [1200000, 2200000]
    ])
    
    print("Raster sampling utilities ready for testing")
    print("To test with actual data:")
    print("1. Place COG files in data/auxiliary/")
    print("2. Use RasterSampler context manager")
    print("3. Call sample_point() or sample_batch()")
