# DC-Trainer: Modular Experimentation Framework

A flexible, modular, and config-driven framework for deep learning training. This project is designed to enable combinatorial experimentation with different encoders, sequence models, and decoders for both Classification and Generation (OCR) tasks.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone github.com/Jayden3316/dc-training.git
cd captcha_ocr

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

Create a `dataset_config.yaml` or use CLI args to generate synthetic data.

```bash
python -m cli generate --width 200 --height 80 --count 1000 --output-dir data/train --word-file data/words.tsv
```

### 3. Run Experiment

Run the training command with your config.

```bash
python -m cli train --config-file experiments/training_configs/generation.yaml
```

---

## Configuration Reference

The framework uses a hierarchical configuration system. A single YAML file (`ExperimentConfig`) controls everything. For a detailed list of all available options, see below.

### 1. Experiment Metadata
Main entry point: `ExperimentConfig`

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `experiment_name` | `str` | `"default_experiment"` | Name of the experiment (used for logging and checkpoints). |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `metadata_path` | `str` | `"validation_set/metadata.json"` | Path to the metadata JSON for the dataset. |
| `image_base_dir` | `str` | `"."` | Base directory where images are located. |
| `train_metadata_path` | `str` | `None` | (Optional) Explicit path to training metadata. |
| `val_metadata_path` | `str` | `None` | (Optional) Explicit path to validation metadata. |

### 2. Dataset Configuration
Main entry point: `DatasetConfig`

#### Basic Settings
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` | `int` | `200` | Width of the generated/input images. |
| `height` | `int` | `80` | Height of the generated/input images. |
| `image_ext` | `str` | `"png"` | Extension for generated images. |

#### Font & Content
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fonts` | `List[str]` | `None` | List of paths to `.ttf` font files. |
| `font_sizes` | `List[int]` | `None` | List of font sizes to pick from. |
| `max_fonts_per_family` | `int` | `2` | Max fonts to pick from each font family folder. |
| `word_path` | `str` | `None` | Path to a TSV file containing words for generation. |
| `word_transform` | `str` | `None` | Key for word transformation (e.g., `"random_capitalize"`). |
| `random_capitalize` | `bool` | `True` | If True, randomly flips case of characters. |

#### Noise & Distortion
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `noise_bg_density` | `int` | `5000` | Number of noise points in the background. |
| `add_noise_dots` | `bool` | `True` | Whether to add small noise dots. |
| `add_noise_curve` | `bool` | `True` | Whether to add a distracting curve. |
| `extra_spacing` | `int` | `-5` | Extra horizontal spacing between characters. |
| `spacing_jitter` | `int` | `6` | Random jitter applied to character spacing. |
| `word_space_probability`| `float`| `0.5` | Probability of inserting a space between words. |
| `word_offset_dx` | `float` | `0.25` | Horizontal offset jitter for the entire word. |

#### Fine-Grained Character Distortion
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `character_offset_dx` | `tuple[int, int]`| `(0, 4)` | Range for random X-offset per character. |
| `character_offset_dy` | `tuple[int, int]`| `(0, 6)` | Range for random Y-offset per character. |
| `character_rotate` | `tuple[int, int]`| `(-30, 30)`| Range for random rotation per character. |
| `character_warp_dx` | `tuple[float, float]`| `(0.1, 0.3)`| Range for horizontal warp factor. |
| `character_warp_dy` | `tuple[float, float]`| `(0.2, 0.3)`| Range for vertical warp factor. |

### 3. Model Architecture
Main entry point: `ModelConfig`

#### Type Selection
| Key | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `encoder_type` | `str` | `"asymmetric_convnext"`| `"asymmetric_convnext"`, `"legacy_cnn"` |
| `projector_type` | `str` | `"linear"` | `"linear"`, `"mlp"`, `"bottleneck"`, `"residual"`, `"identity"` |
| `sequence_model_type`| `str` | `"transformer_encoder"`| `"transformer_encoder"`, `"transformer_decoder"`, `"rnn"`, `"bilstm"` |
| `head_type` | `str` | `"ctc"` | `"linear"`, `"ctc"`, `"mlp"`, `"classification"` |
| `task_type` | `str` | `"generation"` | `"generation"`, `"classification"` |

#### Global Model Dimensions
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | `int` | `256` | Latent dimension size across the pipeline. |
| `d_vocab` | `int` | `62` | Vocabulary size (number of characters). |
| `loss_type` | `str` | `"ctc"` | `"ctc"`, `"cross_entropy"`, `"focal"` |

### 4. Component-Specific Configs

#### Encoder: AsymmetricConvNext
| Key | Default | Description |
| :--- | :--- | :--- |
| `dims` | `[64, 128, 256, 512]`| Channels per stage. |
| `stage_block_counts` | `[2, 2, 6, 2]` | Number of ConvNext blocks per stage. |
| `stem_kernel_size` | `4` | Kernel size for initial stem convolution. |
| `stem_stride` | `4` | Stride for initial stem convolution. |
| `convnext_kernel_size`| `7` | Kernel size in ConvNext blocks. |
| `convnext_drop_path_rate`| `0.0` | Stochastic depth rate. |

#### Encoder: LegacyCNN
| Key | Default | Description |
| :--- | :--- | :--- |
| `filter_sizes` | `[7, 5]` | Kernel sizes per convolution layer. |
| `channels` | `[16, 32]` | Output channels per layer. |
| `patch_height` | `14` | Height of extracted patches. |
| `patch_width` | `7` | Width of extracted patches. |

#### Sequence: Transformer (Encoder/Decoder)
*Inherits from `HookedTransformerConfig`*
| Key | Default | Description |
| :--- | :--- | :--- |
| `n_layers` | `4` | Number of transformer layers. |
| `n_heads` | `8` | Number of attention heads. |
| `d_mlp` | `1024` | Dimension of the MLP sublayer. |
| `n_ctx` | `384` | Maximum sequence length. |
| `dropout` | `0.0` | Global dropout rate. |
| `attn_dropout` | `0.0` | Dropout rate for attention. |

#### Sequence: RNN / BiLSTM
| Key | Default | Description |
| :--- | :--- | :--- |
| `num_layers` | `2` | Number of recurrent layers. |
| `hidden_size` | `256` | Hidden dimension (synced with `d_model`). |
| `dropout` | `0.1` | Dropout rate between layers. |
| `bidirectional` | `False` | (RNN only) Whether to use Bi-directional RNN. |

### 5. Training Configuration
Main entry point: `TrainingConfig`

#### Hyperparameters
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batch_size` | `int" | `32` | Training batch size. |
| `epochs` | `int` | `50` | Total number of training epochs. |
| `learning_rate` | `float` | `1e-4` | Initial learning rate. |
| `optimizer_type` | `str` | `"adamw"` | `"adam"`, `"adamw"`, `"sgd"` |
| `weight_decay` | `float` | `0.01` | L2 regularization factor. |
| `grad_clip_norm` | `float` | `1.0` | Gradient clipping threshold. |
| `accumulate_grad_batches`| `int`| `1` | Number of batches for gradient accumulation. |

#### Logging & Checkpoints
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `checkpoint_dir` | `str` | `"checkpoints"`| Directory to save model checkpoints. |
| `save_every_n_epochs` | `int` | `1` | Frequency of saving checkpoints. |
| `monitor_metric` | `str` | `"val_exact_match"`| Metric used to determine the "best" model. |
| `wandb_project" | `str` | `"captcha-ocr"` | Weights & Biases project name. |
| `log_every_n_steps" | `int` | `10` | Frequency of logging metrics. |

#### Data Loading
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `device` | `str` | `"cuda"` | Execution device (`"cuda"` or `"cpu"`). |
| `num_workers` | `int` | `4` | Number of data loading workers. |
| `mixed_precision` | `bool` | `False` | Whether to use FP16 training. |
| `val_split` | `float` | `0.1` | Fraction of data for validation. |
| `shuffle_train` | `bool` | `True` | Whether to shuffle training data. |

---

## Configuration Examples

### 1. Classification Task
**Architecture**: Asymmetric ConvNeXt -> Projector -> Classification Head (No Transformer).
**Use Case**: Classifying fixed-length captchas or single images into N classes.

```yaml
model_config:
  task_type: "classification"
  
  # 1. Encoder
  encoder_type: "asymmetric_convnext"
  encoder_config:
    dims: [64, 128, 256, 512]
    stage_block_counts: [2, 2, 6, 2]

  # 2. Projector (Encoder Dim -> d_model)
  projector_type: "linear"
  
  # 3. Sequence Model (None / Identity)
  sequence_model_type: null 
  
  # 4. Head (Pools sequence -> Classify)
  head_type: "classification"
  head_config:
    num_classes: 10
    pooling_type: "mean"
    d_model: 256

  d_model: 256
  d_vocab: 10
  loss_type: "cross_entropy"
```

### 2. Generation Task (OCR)
**Architecture**: Asymmetric ConvNeXt -> RNN/BiLSTM -> CTC Head.
**Use Case**: Variable length text recognition (standard Captcha OCR).

```yaml
model_config:
  task_type: "generation"
  
  # 1. Encoder
  encoder_type: "asymmetric_convnext"
  encoder_config:
    dims: [64, 128, 256, 512]
  
  # 2. Projector
  projector_type: "linear"
  
  # 3. Sequence Model (RNN)
  sequence_model_type: "rnn" # or "bilstm"
  sequence_model_config:
    hidden_size: 256 # Matches d_model
    num_layers: 2
    dropout: 0.1
    
  # 4. Head (CTC)
  head_type: "ctc"
  head_config:
    d_model: 256
    d_vocab: 62 # Character set size
    
  d_model: 256
  d_vocab: 62
  loss_type: "ctc"
```
