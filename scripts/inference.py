"""
Run inference on a single NAIP quad to produce a Canopy Height Model (CHM).

Takes a 5-band NAIP DOQQ (R, G, B, NIR, Mask) and uses a trained
UNetFiLM model to generate a CHM. Handles tiling, preprocessing,
model inference, and mosaicking into a final Cloud-Optimized GeoTIFF.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add the project root to the Python path for module resolution
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch

from src.inference_utils import load_model, process_naip_quad, StaticRasterHandler


def setup_logging(output_dir: Path, naip_name: str) -> None:
    """Configure file and console logging."""
    log_file = output_dir / f"{naip_name}_inference.log"
    
    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")


def run_inference(args):
    """
    Orchestrate inference pipeline for a single NAIP quad.
    
    Args:
        args: Parsed command-line arguments.
    """
    start_time = time.monotonic()
    
    # Initialize logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.naip_quad.stem)
    
    logging.info(f"Starting inference for: {args.naip_quad.name}")
    logging.info(f"Chip size: {args.chip_size}x{args.chip_size}")
    logging.info(f"Chip overlap: {args.chip_overlap*100:.0f}%")
    
    if args.dry_run:
        logging.info("DRY RUN MODE: Will validate inputs but not process")
    
    # Static raster paths
    static_dir = args.static_rasters_dir
    elevation_path = static_dir / "elevation.tif"
    climate_pca_path = static_dir / "climate_pca.tif"
    soil_pca_path = static_dir / "soil_pca.tif"
    nlcd_path = static_dir / "nlcd.tif"
    ecoregion_path = static_dir / "ecoregion.tif"
    
    # Verify static rasters
    for raster_path in [elevation_path, climate_pca_path, soil_pca_path, nlcd_path, ecoregion_path]:
        if not raster_path.exists():
            logging.error(f"Static raster not found: {raster_path}")
            sys.exit(1)
    
    logging.info(f"Static rasters directory: {static_dir}")
    
    # Load model
    if not args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        model = load_model(
            checkpoint_path=args.model_checkpoint,
            config_path=args.config,
            device=device
        )
    else:
        model = None
        device = None
    
    # Initialize raster handler
    try:
        with StaticRasterHandler(
            elevation_path=elevation_path,
            climate_pca_path=climate_pca_path,
            soil_pca_path=soil_pca_path,
            nlcd_path=nlcd_path,
            ecoregion_path=ecoregion_path
        ) as raster_handler:
            
            # Process quad
            result = process_naip_quad(
                naip_quad_path=args.naip_quad,
                output_dir=args.output_dir,
                model=model,
                device=device,
                raster_handler=raster_handler,
                chip_size=args.chip_size,
                chip_overlap=args.chip_overlap,
                dry_run=args.dry_run
            )
            
    except Exception as e:
        logging.error(f"Fatal error during inference: {e}")
        sys.exit(1)
    
    # Track runtime
    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    
    result['elapsed_seconds'] = elapsed_time
    result['elapsed_formatted'] = f"{elapsed_time // 60:.0f}m {elapsed_time % 60:.2f}s"
    
    # Write report
    report_file = args.output_dir / f"{args.naip_quad.stem}_report.json"
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logging.info(f"Processing report saved to {report_file}")
    
    # Log results
    if result['status'] == 'success':
        logging.info(f"SUCCESS: Processed {args.naip_quad.name}")
        logging.info(f"  Output: {result['output_path']}")
        logging.info(f"  Chips: {result['processed_chips']}/{result['total_chips']} successful")
        if result['failed_chips'] > 0:
            logging.warning(f"  Failed chips: {result['failed_chips']}")
        logging.info(f"  Time: {result['elapsed_formatted']}")
    elif result['status'] == 'dry_run':
        logging.info(f"DRY RUN COMPLETE: {args.naip_quad.name}")
        logging.info(f"  Dimensions: {result['width']}x{result['height']}")
        logging.info(f"  CRS: {result['crs']}")
        logging.info(f"  DOY: {result['doy']}")
    else:
        logging.error(f"FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def main():
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on a single NAIP quad to produce a Canopy Height Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--naip-quad',
        type=Path,
        required=True,
        help='Path to 5-band NAIP quad GeoTIFF (R,G,B,NIR,Mask).'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save the output Canopy Height Model GeoTIFF.'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-checkpoint',
        type=Path,
        default=Path('model/model.pt'),
        help='Path to trained PyTorch model checkpoint.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to training config YAML.'
    )
    
    # Static rasters
    parser.add_argument(
        '--static-rasters-dir',
        type=Path,
        required=True,
        help='Directory containing static rasters (elevation, climate/soil PCA, nlcd, ecoregion).'
    )
    
    # Processing parameters
    parser.add_argument(
        '--chip-size',
        type=int,
        default=432,
        help='The dimension (width and height) of the square chips for processing. Must be multiple of 16.'
    )
    parser.add_argument(
        '--chip-overlap',
        type=float,
        default=0.2,
        help='Fractional overlap between adjacent chips (e.g., 0.2 for 20%%).'
    )
    
    # Dry run
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and extract metadata without running inference.'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.naip_quad.exists():
        print(f"Error: NAIP quad not found: {args.naip_quad}")
        sys.exit(1)
    
    if not args.dry_run and not args.model_checkpoint.exists():
        print(f"Error: Model checkpoint not found: {args.model_checkpoint}")
        sys.exit(1)
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not args.static_rasters_dir.exists():
        print(f"Error: Static rasters directory not found: {args.static_rasters_dir}")
        sys.exit(1)
    
    if args.chip_size % 16 != 0:
        print(f"Error: Chip size must be multiple of 16, got {args.chip_size}")
        sys.exit(1)
    
    if not 0 < args.chip_overlap < 1:
        print(f"Error: Chip overlap must be between 0 and 1, got {args.chip_overlap}")
        sys.exit(1)
    
    # Run inference
    run_inference(args)


if __name__ == '__main__':
    main()
