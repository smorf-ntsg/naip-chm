"""
Coordinate transformation utilities.
Handles UTM to EPSG:5070 (Albers Equal Area Conic) conversions.
"""

import numpy as np
from pyproj import Transformer
from typing import Tuple, List, Dict
import logging


class CoordinateTransformer:
    """Coordinate transformation with caching."""
    
    def __init__(self):
        self.transformers: Dict[int, Transformer] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_transformer(self, utm_zone: int, target_crs: str = "EPSG:5070") -> Transformer:
        """Get/create transformer for UTM zone."""
        cache_key = (utm_zone, target_crs)
        if cache_key not in self.transformers:
            epsg_utm = f"EPSG:326{utm_zone:02d}"
            
            self.transformers[cache_key] = Transformer.from_crs(
                epsg_utm, 
                target_crs,
                always_xy=True
            )
            self.logger.info(f"Created transformer from {epsg_utm} to {target_crs}")
        
        return self.transformers[cache_key]
    
    def transform_point(self, utm_zone: int, x: float, y: float, target_crs: str = "EPSG:5070") -> Tuple[float, float]:
        """Transform single UTM point."""
        transformer = self.get_transformer(utm_zone, target_crs)
        return transformer.transform(x, y)
    
    def transform_batch(self, utm_zones: np.ndarray, 
                       x_coords: np.ndarray, 
                       y_coords: np.ndarray,
                       target_crs: str = "EPSG:5070") -> np.ndarray:
        """
        Transform batch of UTM coordinates.
        
        Args:
            utm_zones: UTM zone numbers.
            x_coords: UTM x coordinates.
            y_coords: UTM y coordinates.
            target_crs: Target CRS (default EPSG:5070).
            
        Returns:
            Array of shape (n, 2) with transformed coordinates.
        """
        n_points = len(utm_zones)
        transformed_coords = np.zeros((n_points, 2), dtype=np.float64)
        
        unique_zones = np.unique(utm_zones)
        
        for zone in unique_zones:
            mask = utm_zones == zone
            if not np.any(mask):
                continue
                
            transformer = self.get_transformer(zone, target_crs)
            
            x_transformed, y_transformed = transformer.transform(
                x_coords[mask], 
                y_coords[mask]
            )
            
            transformed_coords[mask, 0] = x_transformed
            transformed_coords[mask, 1] = y_transformed
            
            self.logger.debug(f"Transformed {np.sum(mask)} points from UTM zone {zone} to {target_crs}")
        
        return transformed_coords


def transform_coordinates_batch(utm_zones: List[int], 
                              x_coords: List[float], 
                              y_coords: List[float]) -> List[Tuple[float, float]]:
    """
    Batch coordinate transformation helper.
    
    Args:
        utm_zones: List of UTM zone numbers
        x_coords: List of UTM x coordinates  
        y_coords: List of UTM y coordinates
        
    Returns:
        List of (x_5070, y_5070) tuples
    """
    transformer = CoordinateTransformer()
    
    coords_array = transformer.transform_batch(
        np.array(utm_zones),
        np.array(x_coords),
        np.array(y_coords)
    )
    
    return [(x, y) for x, y in coords_array]


def validate_utm_zone(utm_zone: int) -> bool:
    """Validate UTM zone range."""
    return 10 <= utm_zone <= 19


def get_expected_utm_zones() -> List[int]:
    """Get expected UTM zones."""
    return list(range(12, 19))  # Zones 12-18


if __name__ == "__main__":
    # Test coordinate transformation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    transformer = CoordinateTransformer()
    
    # Test single point transformation
    x_5070, y_5070 = transformer.transform_point(16, 638594, 3377628)
    print(f"UTM 16: (638594, 3377628) -> EPSG:5070: ({x_5070:.2f}, {y_5070:.2f})")
    
    # Test batch transformation
    test_zones = [16, 17, 16]
    test_x = [638594, 278186, 629263]
    test_y = [3377628, 4339026, 4919514]
    
    coords_5070 = transformer.transform_batch(
        np.array(test_zones),
        np.array(test_x),
        np.array(test_y)
    )
    
    print("\nBatch transformation results:")
    for i, (zone, x, y) in enumerate(zip(test_zones, test_x, test_y)):
        x_5070, y_5070 = coords_5070[i]
        print(f"UTM {zone}: ({x}, {y}) -> EPSG:5070: ({x_5070:.2f}, {y_5070:.2f})")
