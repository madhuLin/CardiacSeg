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

## Advanced Scripts Usage

For more fine-grained control over the pipeline, you can use the following scripts. These scripts are designed to work with the `experiments_config.py` file, which acts as a central hub for defining and managing different model configurations.

### `experiments_config.py`

This file is not a script to be executed, but a configuration file that defines the hyperparameters for each experiment. Before running the scripts below, you should review and edit this file to enable the models you want to work with and adjust their parameters.

```python
EXPERIMENTS = {
    "AICUP_attention_unet": {
        "enabled": True,  # Set to True to run this model
        "vote": True,     # Set to True to include in ensemble voting
        "model_name": "attention_unet",
        "params": {
            "roi_z": 112,
            "lr": 1e-4,
            # ... other parameters
        },
    },
    # ... other experiments
}
```

### `gen_kfold_splits.py`

This script generates JSON files for k-fold cross-validation. It takes a main data dictionary JSON and splits the training set into `k` folds, creating `k` new JSON files.

**Usage:**

```bash
python gen_kfold_splits.py \
  --workspace_dir /path/to/your/CardiacSegV2 \
  --input_json AICUP_training_50.json \
  --kfold 5
```

This will create `AICUP_training_50_fold1.json`, `..._fold2.json`, etc., in the `CardiacSegV2/exps/data_dicts/chgh/` directory.

### `train_k.py` (and `run_all_models.py`)

`train_k.py` is the primary, recommended script for running training sessions. It reads experiment settings from `experiments_config.py` and can train one or more models, with full support for k-fold cross-validation.

The older `run_all_models.py` script provides similar functionality but **lacks k-fold support**. It is recommended to use `train_k.py` for all training tasks, as it is more versatile.

**Usage:**

-   **Train specific models across all 5 folds:**

    ```bash
    python train_k.py \
      --exp_ids AICUP_attention_unet AICUP_dynunet \
      --kfold 5
    ```

-   **Train all `enabled=True` models for a single run (equivalent to `run_all_models.py`):**

    ```bash
    python train_k.py --kfold 1
    ```
    *(Note: Using `--kfold 1` or omitting it trains on a single data split, making it behave like the older `run_all_models.py` script.)*

Trained models and logs will be saved under `models/<exp_name>_f<fold_number>/` and `logs_cli/<exp_name>_f<fold_number>.log`.

### `run_infer_config.py`

After training, this script runs inference on a directory of images using the models defined in `experiments_config.py`. It will automatically find the best checkpoint for each specified experiment.

**Usage:**

```bash
python run_infer_config.py \
  --images_dir /path/to/your/nifti_images \
  --exp_ids AICUP_attention_unet_f1 AICUP_dynunet_f1
```

- If `--exp_ids` is not provided, it will run inference for all models where `"vote": True` is set in the config.
- Inference results are saved to `myo_pred/<data_name>/infer/<exp_id>/`.

### `run_ensemble_vote.py`

This script performs voxel-wise voting to ensemble the predictions from multiple models, which can improve segmentation accuracy.

**Usage:**

-   **Majority Vote:**
    Combines predictions from several models with equal weight.

    ```bash
    python run_ensemble_vote.py \
      --exp_ids AICUP_attention_unet_f1 AICUP_dynunet_f1 AICUP_swinunetr_f1 \
      --ensemble_name my_3_models_ensemble
    ```

-   **Weighted Vote:**
    You can assign weights to models (e.g., based on their validation Dice scores) and also apply weights to specific classes (e.g., giving more importance to the myocardium).

    ```bash
    python run_ensemble_vote.py \
      --exp_ids AICUP_attention_unet_f1 AICUP_dynunet_f1 \
      --weight AICUP_attention_unet_f1=0.85 --weight AICUP_dynunet_f1=0.83 \
      --class_weights 1.0 1.0 2.0 1.0 \
      --ensemble_name weighted_myo_focus
    ```
The output is a new set of segmentation files saved in `myo_pred/<data_name>/infer/ensemble_<ensemble_name>/`.
