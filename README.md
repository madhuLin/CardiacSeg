# CardiacSeg

A deep learning pipeline for 3D medical image segmentation, with a focus on cardiac structures. This project is built using PyTorch and MONAI.

## Features

- **Multiple SOTA Models:** Implements various 3D segmentation models, including UNet, Attention UNet, VNet, UNETR, SwinUNETR, and more.
- **End-to-End Pipeline:** Provides a complete pipeline from training to inference and evaluation.
- **Reproducibility:** Training and inference can be executed through simple command-line scripts with preset configurations.
- **K-Fold Cross-Validation:** Supports k-fold cross-validation for robust evaluation.
- **Ensemble-Ready:** Includes scripts to perform ensemble voting from multiple trained models.

## Acknowledgements

This project is heavily based on the [MONAI](https://monai.io/) framework. Many of the neural network architectures are either taken directly from or inspired by MONAI's implementations.
Additionally, this codebase references concepts and implementations from the following Google Colab notebook: [https://colab.research.google.com/drive/1iC7i_EWCZsCr5T-7jDD77V8dt_simGsn?usp=sharing](https://colab.research.google.com/drive/1iC7i_EWCZsCr5T-7jDD77V8dt_simGsn?usp=sharing)

The following model architectures are included, and we acknowledge the original authors for their contributions to the field:
- **UNETR**: [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)
- **SwinUNETR**: [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)
- **CoTr**: [CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.03024)
- **UX-Net**: [UX-Net: A 3D-CNN based on an Unbalanced U-shaped Network for Medical Image Segmentation](https://arxiv.org/abs/2203.04037)

## Project Structure

```
.
├── CardiacSegV2/
│   ├── data_utils/       # Data loading and processing utilities
│   ├── datasets/         # Dataset-specific classes
│   ├── expers/           # Core experiment scripts (train, test, tune)
│   ├── networks/         # Model architectures
│   ├── optimizers/       # Optimizers and LR schedulers
│   ├── runners/          # Training, testing, and inference runners
│   └── transforms/       # Data augmentation and pre-processing
├── experiments_config.py # Configuration file for experiments
├── requirements.txt      # Python dependencies
│
├── run.py                # Main script for training with presets
├── train_k.py            # Script for k-fold cross-validation training
├── unetr_run.py          # Script to run UNETR training with various configs
│
├── infer_batch.py        # Generic script for batch inference
├── infer_k.py            # Script for inference with k-fold models
├── run_infer_config.py   # Inference script using experiment configs
├── unetr_infer.py        # Inference script for UNETR models
│
├── run_ensemble_vote.py  # Script to perform ensemble voting
└── README.md             # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd CardiacSeg
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project uses scripts in the root directory to handle training and inference.

### Training

The easiest way to start training is to use the `run.py` script with a preset configuration.

1.  **Choose a preset:** Open the `run.py` file and look at the `TRAIN_PRESETS` dictionary to see the available options (e.g., `unet3d_f48`, `unetr_f64_d3-4-10`, `swinunetr_small`).

2.  **Run training:**
    Execute the script with your chosen preset. The script will handle all the necessary parameters for the training pipeline.

    ```bash
    # Example: Train a UNETR model with a specific configuration
    python run.py --preset unetr_f64_d3-4-10
    ```
    
    Trained models will be saved in the `models/<exp_name>/` directory.

### Inference

Once you have a trained model, you can use one of the inference scripts to generate segmentations on new images.

1.  **Place your images:**
    Put the NIfTI images you want to process into a directory. For example: `myo_pred/chgh/image/`.

2.  **Run inference:**
    Use the `unetr_infer.py` script for UNETR models or `infer_batch.py` for other models. You need to specify the preset or experiment name that corresponds to your trained model.

    ```bash
    # Example: Run inference with a trained UNETR model
    python unetr_infer.py --preset unetr_f64_d3-4-10-3_lr5e-5 --images_dir myo_pred/chgh/image/
    ```

    The segmentation results will be saved in the `myo_pred/chgh/infer/<preset_name>/` directory.
