# DC-Trainer: Modular Experimentation Framework

A flexible, modular, and config-driven framework for deep learning training, designed for combinatorial experimentation with different encoders, sequence models, and decoders for both **Classification** (ResNet-like) and **Generation/OCR** (CRNN/Transformer) tasks.

[**Read the Full Documentation**](DOCUMENTATION.md)

## Features
*   **Modular Architecture**: Plug-and-play Encoders (ConvNext, ResNet), Adapters, Projectors, Sequence Models (Transformer, RNN), and Heads (CTC, Classification).
*   **Config-Driven**: Strictly typed, hierarchical configuration system using YAML.
*   **Multi-Task**: Supports both fixed-label classification and variable-length sequence generation.
*   **Production Ready**: Includes training, evaluation, and inference scripts.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Jayden3316/dc-training.git
cd captcha_ocr

# Install dependencies (Python 3.8+)
pip install -r requirements.txt
```

### 2. Generate Data
Generate a synthetic dataset using a config file.

```bash
python -m cli generate --config-file experiments/dataset_configs/default.yaml --out-dir data/train --dataset-count 1000
```

### 3. Run Experiment
Train a model using an experiment configuration.

```bash
python -m cli train --config-file experiments/training_configs/generation.yaml
```

### 4. Evaluate & Inference
Evaluate on a validation set or run inference on new images.

```bash
# Evaluate
python -m cli evaluate --checkpoint checkpoints/best_model.pth --metadata-path data/val/metadata.json

# Inference
python -m cli inference --checkpoint checkpoints/best_model.pth --image-paths test_image.png
```

---

## Configuration & Architecture

The framework uses a pipeline approach:
`Input -> Encoder -> Adapter -> Projector -> Sequence Model -> Head -> Output`

For detailed configuration options and architecture diagrams, please see the [**Documentation**](DOCUMENTATION.md).
