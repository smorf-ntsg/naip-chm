# NAIP-CHM: A 0.6-meter Resolution Canopy Height Model for the Contiguous United States

## Overview

This repository contains the source code, trained model weights, and inference tools for **NAIP-CHM**, a project that generates a 0.6-meter resolution canopy height model (CHM) for the contiguous United States using National Agriculture Imagery Program (NAIP) aerial imagery.

This codebase supports the upcoming paper: **A 0.6-meter resolution canopy height model for the contiguous United States**.

The repository provides:
*   **Inference Pipeline:** Tools to generate canopy height models from NAIP DOQQs using a pre-trained U-Net model.
*   **Training Code:** Scripts to train the model on new data, including distributed training support.
*   **Pre-trained Model:** The final model weights used to generate the CONUS-wide dataset.
*   **Conditioning Data:** Static environmental raster data required for model inference.

## Installation

This codebase requires **Python 3.11+**.

1.  Ensure you have the repository files extracted to your local machine.

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Data Access

### 1. Training Dataset
The training dataset is not included in this repository due to its size. It contains over 22 million image pairs.
You can download the training data from the University of Montana Rangeland server:
*   **URL:** http://rangeland.ntsg.umt.edu/data/rap/chm-naip/

See `data/training_dataset/README` for more details.

### 2. Inference Assets
This repository includes the necessary assets to run inference:
*   **Pre-trained Model Weights:** Located at `model/model_20251016.pt`
*   **Static Conditioning Rasters:** Located in `data/conditioning_data/` (includes elevation, climate, soil, NLCD, and ecoregion data).

---

## Usage

### Running Inference
You can run the model on a standard NAIP DOQQ (Digital Ortho Quarter Quad) using the `scripts/inference.py` script. A sample NAIP image is provided in `data/naip_doqqs/`.

**Example Command:**
```bash
python scripts/inference.py \
  --naip-quad data/naip_doqqs/m_3812259_nw_10_060_20220519.tif \
  --output-dir output/ \
  --model-checkpoint model/model_20251016.pt \
  --config configs/config.yaml \
  --static-rasters-dir data/conditioning_data/
```

**Arguments:**
*   `--naip-quad`: Path to the input 4-band NAIP imagery (R, G, B, NIR).
*   `--output-dir`: Directory where the output CHM GeoTIFF and report will be saved.
*   `--model-checkpoint`: Path to the trained model weights.
*   `--config`: Path to the configuration YAML file.
*   `--static-rasters-dir`: Directory containing the environmental conditioning rasters.
*   `--chip-size`: (Optional) Processing chip size (default: 432).
*   `--chip-overlap`: (Optional) Overlap between chips (default: 0.2).

### Training the Model
To train the model from scratch or fine-tune it, use the `scripts/train.py` script. Ensure you have downloaded the training dataset and updated the `configs/config.yaml` file to point to the correct data paths.

**Command:**
```bash
python scripts/train.py \
  --config configs/config.yaml \
  --experiment_name my_experiment
```

**Distributed Training:**
The script supports Distributed Data Parallel (DDP) training. To run on multiple GPUs (e.g., 2 GPUs):
```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/config.yaml \
  --experiment_name my_ddp_experiment
```

---

## Citation

If you use this dataset or code in your research, please cite the following manuscript:

> Morford, S. L., Allred, B. W., Coons, S. P., Marcozzi, A. A., McCord, S. E., & Naugle, D. E. (2025). A 0.6-meter resolution canopy height model for the contiguous United States. *[Journal Name TBD]*.

**Training Dataset Reference:**
> Allred, B. W., McCord, S. E. & Morford, S. L. Canopy height model and NAIP imagery pairs across CONUS. *Sci. Data* 12, 322 (2025).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
