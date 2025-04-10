# COMPASS: Computational Mapping of Predicted Active Subcellular Metabolites
## Project Overview

COMPASS is a deep learning framework for prediction of dynamic metaboltes derived from different nutrients using spontaneous Raman and DO-SRS, specifically designed for spatial distribution prediction in hyperspectral images (HSI).

## Key Features
- Advanced preprocessing utilities to reduce the distribution gap between training data and real application.
- Spectral data training using 1D-ResNet architecture
- Training and validation pipelines
- Transfer learning capabilities via model fine-tuning
- Hyperspectral image (HSI) analysis for spatial metabolites distribution prediction
- Visualization tools for model performance and representation learning interpretation


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Zhi-Li-SRS/COMPASS.git
   cd COMPASS
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

To train the model:

```bash
python train.py --train_data_path Raman_dataset/lipids/train_data_with_bg.csv \
                --val_data_path Raman_dataset/lipids/val_data_with_bg.csv \
                --checkpoint_dir checkpoints_new \
                --num_classes 18 \
                --batch_size 512 \
                --epochs 200
```

### Fine-tuning a Pre-trained Model

To fine-tune a pre-trained model with additional classes (e.g., adding a background class):

```bash
python finetune.py --pretrained_model_path checkpoints_pretrain/best_model.pth \
                   --train_data_path Raman_dataset/molecules_9/train_data.csv \
                   --val_data_path Raman_dataset/molecules_9/val_data.csv \
                   --checkpoint_dir checkpoints_lipids_and_protein
```

### Inference on Hyperspectral Images

To predict molecular distributions in hyperspectral images:

```bash
python hsi_infer.py --model_path checkpoints_with_bg/best_model.pth \
                    --library_path Raman_dataset/library.csv \
                    --image_path HSI_data/1-Plin1_FB.tif \
                    --output_dir prediction_results/hsi_plin1_with_bg \
                    --target d2-fructose d7-glucose d-tyrosine d-methionine d-leucine
```

For supervised inference on specific targets:

```bash
python hsi_infer_supervised --model_path checkpoints_lipids_and_protein/best_model.pth \
                           --library_path Raman_dataset/molecules_9/train_reference.csv \
                           --image_path HSI_data/1-Wt_FB.tif \
                           --output_dir prediction_results/1-Wt
```

### Model Architecture

COMPASS uses a 1D ResNet architecture with the following components:
- Convolutional layers for feature extraction
- Residual blocks for deep structure and gradient propagation
- Batch normalization and dropout for regularization
- Adaptive pooling and fully connected layers for classification

## Data Preparation

The project expects CSV files with the following format:
- First column: Class/molecule name
- Remaining columns: Spectral intensity values at corresponding wavenumbers

Preprocessing functions in `utils.py` and `preprocess.py`


## Visualization

The project includes several visualization tools:
- Training curves and confusion matrices (`vis_utils.py`)
- t-SNE embeddings of extracted features (`Raman_inference.py`)
- Grad-CAM visualizations to interpret model decisions on library data


## License

This project is licensed under the terms included in the LICENSE file.
