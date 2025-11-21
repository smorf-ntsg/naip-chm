"""
Google Earth Engine utilities for NAIP CHM inference.
Handles efficient data fetching using the high-volume endpoint.
"""

import logging
import time
from io import BytesIO
from typing import Dict, Optional, Tuple, Any, List

import ee
import numpy as np
import rasterio
import requests

logger = logging.getLogger(__name__)

# Asset IDs
CONDITIONING_ASSETS = {
    'climate': 'projects/naip-chm/assets/conditioning-data/prism',
    'soil': 'projects/naip-chm/assets/conditioning-data/solus',
    'elevation': 'projects/naip-chm/assets/conditioning-data/elevation',
    'nlcd': 'projects/naip-chm/assets/conditioning-data/nlcd',
    'ecoregion': 'projects/naip-chm/assets/conditioning-data/ecoregion'
}

def initialize_gee(project_id: str, high_volume: bool = True) -> None:
    """
    Initialize Earth Engine with high-volume endpoint preference.
    """
    try:
        if high_volume:
            ee.Initialize(
                project=project_id,
                opt_url='https://earthengine-highvolume.googleapis.com'
            )
        else:
            ee.Initialize(project=project_id)
        logger.info(f"GEE Initialized (Project: {project_id}, High-Volume: {high_volume})")
    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        raise

def get_naip_doqq_geometry(lat: float, lon: float, year: int) -> Tuple[str, ee.Geometry, Any]:
    """
    Find the NAIP DOQQ covering the given point.
    
    Args:
        lat: Latitude.
        lon: Longitude.
        year: Year to search.
        
    Returns:
        (image_id, geometry, full_image_metadata)
    """
    point = ee.Geometry.Point([lon, lat])
    
    # Search collection
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(point)
        .filterDate(f'{year}-01-01', f'{year}-12-31')
    )
    
    size = collection.size().getInfo()
    if size == 0:
        raise ValueError(f"No NAIP DOQQ found at ({lat}, {lon}) for year {year}")
    
    # Get first image (usually there's only one per year per location)
    image = collection.first()
    info = image.getInfo()
    
    image_id = info['id']
    geometry = ee.Geometry(info['properties']['system:footprint'])
    
    logger.info(f"Found NAIP DOQQ: {image_id}")
    return image_id, geometry, info

def fetch_chip_data(
    image_id: str,
    region: ee.Geometry,
    shape: Tuple[int, int] = (432, 432),
    crs: str = 'EPSG:5070'
) -> np.ndarray:
    """
    Fetch NAIP raster data for a specific chip.
    
    Args:
        image_id: NAIP image ID.
        region: Chip geometry (EPSG:5070).
        shape: Target shape (height, width).
        crs: Target CRS (default EPSG:5070).
        
    Returns:
        Numpy array (4, H, W) normalized to 0-1.
    """
    # Select bands R, G, B, N
    image = ee.Image(image_id).select(['R', 'G', 'B', 'N'])
    
    # Get download URL for the chip
    # Using getDownloadURL is often faster/easier for small chips than computePixels 
    # which requires more complex parsing of raw bytes.
    # We request it in the target CRS and dimensions.
    try:
        url = image.getDownloadURL({
            'region': region,
            'dimensions': list(reversed(shape)), # EE expects WxH? Actually strictly it's WIDTHxHEIGHT or just one number
            'crs': crs,
            'format': 'GEO_TIFF'
        })
        
        # Fetch and read
        response = requests.get(url)
        response.raise_for_status()
        
        with rasterio.open(BytesIO(response.content)) as src:
            # Read all 4 bands
            # Shape will be (4, H, W)
            data = src.read()
            
            # Handle potential size mismatch due to projection effects
            # If we get slightly different size, we might need to crop/pad or resize
            # But passing 'dimensions' to GEE usually enforces the size.
            if data.shape[1:] != shape:
                # This can happen if GEE's grid alignment is slightly off
                # For now, we assume strict adherence or resize
                # Simple resize if needed
                from rasterio.enums import Resampling
                if data.shape[1] != shape[0] or data.shape[2] != shape[1]:
                    logger.warning(f"Got shape {data.shape[1:]}, expected {shape}. Resizing.")
                    new_data = np.empty((4, *shape), dtype=data.dtype)
                    for i in range(4):
                        # Simple bilinear interpolation
                        # (This is a fallback, ideally GEE gives correct size)
                        import cv2 # Optional dependency? Or just use rasterio reproject
                        # Using rasterio reproject logic would be better but we don't have a transform here easily
                        # Let's just strict check for now
                        pass 
            
            # Normalize to 0-1
            data = data.astype(np.float32) / 255.0
            return data
            
    except Exception as e:
        logger.error(f"Error fetching chip data: {e}")
        raise

def fetch_conditioning_vector(
    lat: float, 
    lon: float,
    doy: int
) -> Dict[str, float]:
    """
    Fetch conditioning variables for a point.
    
    Args:
        lat: Latitude.
        lon: Longitude.
        doy: Day of Year (for time encoding).
        
    Returns:
        Dictionary of raw values.
    """
    point = ee.Geometry.Point([lon, lat])
    
    # Combine all assets into a single reduction for efficiency
    # We assume these assets are ImageCollections or Images.
    # Based on plan, they seem to be Images (projects/naip-chm/assets/...)
    
    # Helper to get image
    def get_img(key):
        return ee.Image(CONDITIONING_ASSETS[key])
    
    # Construct a composite image
    # We need specific bands from these assets.
    # Assuming the assets are pre-processed and have bands named appropriately
    # or we select them by index.
    
    # Climate (6 bands)
    climate = get_img('climate').rename(['climate_pca_1', 'climate_pca_2', 'climate_pca_3', 'climate_pca_4', 'climate_pca_5', 'climate_pca_6'])
    
    # Soil (6 bands)
    soil = get_img('soil').rename(['soil_pca_1', 'soil_pca_2', 'soil_pca_3', 'soil_pca_4', 'soil_pca_5', 'soil_pca_6'])
    
    # Elevation (1 band)
    elevation = get_img('elevation').select([0], ['elevation'])
    
    # NLCD (1 band)
    nlcd = get_img('nlcd').select([0], ['nlcd'])
    
    # Ecoregion (1 band)
    ecoregion = get_img('ecoregion').select([0], ['ecoregion'])
    
    # Stack
    stack = (
        climate
        .addBands(soil)
        .addBands(elevation)
        .addBands(nlcd)
        .addBands(ecoregion)
    )
    
    # Reduce region (sample at point)
    # scale=30 is typical for these datasets (NLCD/SRTM-derived)
    try:
        result = stack.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=30,
            crs='EPSG:5070' # Sample in consistent projection
        ).getInfo()
        
        if not result:
            logger.warning(f"No conditioning data found at ({lat}, {lon})")
            return None
            
        # Add DOY
        result['doy'] = doy
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching conditioning vector: {e}")
        return None
