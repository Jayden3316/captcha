# Experiments for Captcha OCR

Tracks all experiments performed with paths to their configs, wandb logs, kaggle notebook version number for runs and results.

The motivation behind the set of experiments is to try minimize the number of factors that affect a result in an individual run. 
- The choice of the image encoder size is determined through experiments on fixed datasets so that downstream effects like pooling or sequence models being inadequate are not a factor.
- The codebase currently supports two pipelines for image classification: standard_classification and sequence_classification. See DOCUMENTATION.md for a detailed description. The adapter type is fixed to flatten for standard_classification and vertical_feature (often followed by a projector) for sequence_classification.
- Training config is largely the same everywhere, and has been reproduced here for reference:

For pre-generated datasets (and this config is set up for classification tasks):

```yaml
training_config:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  optimizer_type: "adamw"
  weight_decay: 0.01
  grad_clip_norm: 1.0
  accumulate_grad_batches: 1
  
  checkpoint_dir: "experiments/<task>/<experiment_name>/checkpoints" # updated as needed
  save_every_n_epochs: 1
  monitor_metric: "val_acc"
  wandb_project: "captcha-ocr"
  log_every_n_steps: 10
  
  device: "cuda"
  num_workers: 4
  mixed_precision: false
  shuffle_train: true

  metrics: ['accuracy', 'f1', 'precision', 'recall']
```

For on-the-fly generated datasets (and this config is set up for generation tasks):

```yaml
training_config:
  batch_size: 32
  
  training_steps: 50000 # num_samples = training_steps * batch_size = 1.6 * 10^6 
  # training_steps is set to be of the order of the number of parameters in the model
  # should be adjusted for every model
  # for generation tasks, the number of possible words are huge
  # so the assumption is that the problem is model constrained.

  use_onthefly_generation: True
  save_every_steps: 2048
  val_check_interval_steps: 2048
  val_steps: 64

  learning_rate: 0.0001
  optimizer_type: "adamw"
  weight_decay: 0.01
  grad_clip_norm: 1.0
  accumulate_grad_batches: 1
  
  checkpoint_dir: "experiments/<task>/<experiment_name>/checkpoints" # updated as needed


  save_every_n_epochs: 1
  monitor_metric: "val_exact_match"

  wandb_project: "captcha-ocr"
  # wandb_name: "<experiment_name>" Takes experiment from config file if not specified
  log_every_n_steps: 10
  
  device: "cuda"
  num_workers: 4
  mixed_precision: false
  shuffle_train: true

  metrics: ['character_accuracy', 'word_correct', 'edit_distance']
```
## Datasets
Multiple datasets of various sizes could be used. 

The config for noisy dataset generation is provided here for reference:
noisy_dataset_train:

```yaml
dataset_config:
    width: 192
    height: 64
    width_divisor: 16
    width_bias: 0
    resize_mode: "variable"
    image_ext: "png"

    fonts_root: "./train_font_library" 
    fonts: []
    max_fonts_per_family: 1 
    font_sizes: [42]

    random_capitalize: True
    add_noise_dots: True
    add_noise_curve: True

    noise_bg_density: 5000

    extra_spacing: -5
    spacing_jitter: 5
    word_space_probability: 0.5
    word_offset_dx: 0.25

    character_offset_dx: [0, 4]
    character_offset_dy: [0, 6]
    character_rotate: [-30, 30]
    character_warp_dx: [0.1, 0.3]
    character_warp_dy: [0.2, 0.3]

    # random colors are used by default
    # bg_color: [255, 255, 255, 1]
    # fg_color: [0, 0, 0]

    vocab: '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    min_word_len: 4
    max_word_len: 10
```

These are some exemplar configurations to easily refer to when needed during discussion of experiments below.

| Dataset name | Dataset Config | words list | font_root | random_capitalize | remarks |
| --- | --- | --- | --- | --- | --- |
| **fixed_clean_dataset_train_classification** | experiments/dataset_configs/clean_dataset_train.yaml | experiments/diverse_words.tsv | train_font_library | False |clean dataset, 5 letter alphanumeric words, no noise, no distortions, black text on white background |
| **fixed_clean_dataset_val_classification** | experiments/dataset_configs/clean_dataset_val.yaml | experiments/diverse_words.tsv | val_font_library | False | clean dataset, 5 letter alphanumeric words, no noise, no distortions, black text on white background |
| **fixed_noisy_dataset_train_classification** | experiments/dataset_configs/noisy_dataset_train.yaml | experiments/diverse_words.tsv | train_font_library | False | noisy dataset, 5 letter alphanumeric words, noise, distortions, randomly colored text on a light background |
| **fixed_noisy_dataset_val_classification** | experiments/dataset_configs/noisy_dataset_val.yaml | experiments/diverse_words.tsv | val_font_library | False | noisy dataset, 5 letter alphanumeric words, noise, distortions, randomly colored text on a light background |

TODO: list other relevant datasets here

These datasets enable comparison of the following:
- Effect of dataset size on training (for fixed and variable datasets) and (noisy and clean datasets)
- Effect of fixed or variable datasets
- Effect of noisy or clean datasets

## Classification

### Determining the size of model

All of the following experiments are performed on `fixed_noisy_train` with `standard_classification` pipeline.

| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| resnet_small | experiments/training_configs/classification/resnet_small.yaml | | | | |
| resnet_base | experiments/training_configs/classification/resnet_base.yaml | | | | |
| resnet_large | experiments/training_configs/classification/resnet_large.yaml | | | | |
| --- | --- | --- | --- | --- | --- |

### Comparing architectures

There are primarily two kinds of experiments in this subsection:
- Comparing between ConvNext and ResNet
- Setting a baseline performance of using a `SequenceModel` after the image encoder.

#### Architectural considerations:

Since captchas are of variable width, handling them seem to have a few variations:
- Resizing to a fixed width
- Pooling the feature map along the spatial dimensions
- Using a sequence model to process the feature map

Pooling would result in a loss of spatial information, but this could be handled if there are a sufficient number of channels.

Resizing to a fixed width would harm longer captchas and compressing these might not be optimal. Finally, some sort of tiling/ sliding window approach might be possible for generation tasks, but implementing them for classification without pooling seems more complex.

There are previous work that use a sequence model after an encoder like CRNN. Here, the feature maps from the ImageEncoder are reshaped and passed as visual tokens. The sequence model then processes these visual tokens to generate the final output.

The effective stride along the width is determined by the embedding dimension since `VerticalFeatureAdapter` does the following transform: `[B, C, H, W] -> [B, W//f, C * H * f], where f = (output_dim // C * H)`. 

By default, both image encoders use the following pattern:

- Stem: `[B, C, H, W] -> [B, C', H/4, W/4]`
- Block1: `[B, C', H/4, W/4] -> [B, C', H/4, W/4]`
- Downsample1: `[B, C', H/4, W/4] -> [B, C'', H/8, W/8]`
- Block2: `[B, C'', H/8, W/8] -> [B, C'', H/8, W/8]`
- Downsample2: `[B, C'', H/8, W/8] -> [B, C''', H/16, W/16]`
- Block3: `[B, C''', H/16, W/16] -> [B, C''', H/16, W/16]`
- Downsample3: `[B, C''', H/16, W/16] -> [B, C'''' H/32, W/32]`
- Block4: `[B, C'''' H/32, W/32] -> [B, C'''' H/32, W/32]`

This means that each vision token covers a stride of at least 32px along the width and the entire height. Since the width of the characters are also approximately 20-40px, each token could cover multiple characters, especially if 'narrow' letters appear together. 

We provide the model configs here for reference:
resnet-base:

```yaml
model_config:
    pipeline_type: "standard_classification"
    task_type: "classification"
    encoder_type: "resnet"
    encoder_config:
        dims: [8, 16, 24, 48]
        stem_kernel_size: 4
        stem_stride: 4
        stem_in_channels: 3
        stage_block_counts: [2, 2, 6, 2]
        downsample_strides: [(2, 1), (2, 2), (2, 2)]
        downsample_kernels: [(2, 1), (2, 2), (2, 2)]
        downsample_padding: [(0, 0), (0, 0), (0, 0)]

    adapter_type: "flatten"
    adapter_config:
        output_dim: 576 # 48 * 2 * 6
    head_type: "classification"
    head_config:
        num_classes: 100
        d_model: 576
        head_hidden_dim: 256
        pooling_type: "mean" # default arg, not used here.
    
    # global config
    d_model: 576
    d_vocab: 100
    loss_type: "cross_entropy"

```
resnet-base-rnn: 

```yaml
model_config: 
    pipeline_type: "sequence_classification"
    task_type: "classification"
    encoder_type: "resnet"
    encoder_config: 
        dims: [16, 32, 64, 128]
        stem_kernel_size: 4
        stem_stride: 4
        stem_in_channels: 3
        stage_block_counts: [2, 2, 6, 2]
        downsample_strides: [(2, 2), (2, 1), (2, 1)]
        downsample_kernels: [(2, 2), (2, 3), (2, 3)]
        downsample_padding: [(0, 0), (0, 1), (0, 1)]
        # ([B, C, H, W] -> [B, C', H/4, W/4]) -> ([B, C'', H/8, W/8] -> [B, C''', H/16, W/8] -> [B, C'''', H/32, W/8]) 
        # Effective stride along the horizontal axis is 8 

    adapter_type: "vertical_feature"
    adapter_config:
        output_dim: 256 # f * C * H = f * 128 * 2; f = 1
        # using f = 1 sets the sequence length to be W/8, which is between 20-40 tokens for most captchas
    sequence_model_type: 'rnn'
    sequence_model_config:
        hidden_size: 256
        num_layers: 2
        dropout: 0.1
        bidirectional: False

    sequence_adapter_type: "sequence_pool"
    sequence_adapter_config:
        pool_type: "last" # Use last hidden state for RNN

    head_type: "classification"
    head_config:
        num_classes: 100
        d_model: 256
        head_hidden_dim: 192
        pooling_type: "mean" # default arg, not used here.
    
    # global config
    d_model: 256
    d_vocab: 100
    loss_type: "cross_entropy"
```
finally, another alternative is to have (2,1) kernels instead of (2,3) kernels, with stride as (2, 1) and no padding. The tradeoff is that there would be no context from the adjecent patches during downsampling, but the performance difference is not entirely obvious to me.

resnet-base-rnn-narrow-asymm:

The config is provided here for reference:
```yaml
model_config:
    pipeline_type: "standard_classification"
    task_type: "classification"
    encoder_type: "resnet"
    encoder_config:
        dims: [16, 32, 64, 128]
        stem_kernel_size: 4
        stem_stride: 4
        stem_in_channels: 3
        stage_block_counts: [2, 2, 6, 2]
        downsample_strides: [(2, 2), (2, 1), (2, 1)]
        downsample_kernels: [(2, 2), (2, 1), (2, 1)]
        downsample_padding: [(0, 0), (0, 0), (0, 0)]

        # the following are defaults, but reproduced here for clarity:
        convnext_kernel_size: 7
        convnext_drop_path__rate: 0.0
        convnext_expansion_ratio: 4

    adapter_type: "flatten"
    adapter_config:
        output_dim: 576 # 128 * 2 * 6
    head_type: "classification"
    head_config:
        num_classes: 100
        d_model: 576
        head_hidden_dim: 256
        pooling_type: "mean" # default arg, not used here.
    
    # global config

```
| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| resnet_base | experiments/training_configs/classification/resnet_base.yaml | | | | |
| resnet_base_rnn | experiments/training_configs/classification/resnet_base_rnn.yaml | | | | |
| resnet_base_rnn_narrow_asymm | experiments/training_configs/classification/resnet_base_rnn_narrow_asymm.yaml | | | | |

#### ConvNext vs ResNet

Experiments are done on `fixed_noisy_train` with `standard_classification` pipeline.

The ConvNext config is provided here for reference:
```yaml
model_config:
    pipeline_type: "standard_classification"
    task_type: "classification"
    encoder_type: "convnext"
    encoder_config:
        dims: [16, 32, 64, 128]
        stem_kernel_size: 4
        stem_stride: 4
        stem_in_channels: 3
        stage_block_counts: [2, 2, 6, 2]
        downsample_strides: [(2, 2), (2, 2), (2, 2)]
        downsample_kernels: [(2, 2), (2, 2), (2, 2)]
        downsample_padding: [(0, 0), (0, 0), (0, 0)]

        # the following are defaults, but reproduced here for clarity:
        convnext_kernel_size: 7
        convnext_drop_path__rate: 0.0
        convnext_expansion_ratio: 4

    adapter_type: "flatten"
    adapter_config:
        output_dim: 576 # 128 * 2 * 6
    head_type: "classification"
    head_config:
        num_classes: 100
        d_model: 576
        head_hidden_dim: 256
        pooling_type: "mean" # default arg, not used here.
    
    # global config
    d_model: 576
    d_vocab: 100
    loss_type: "cross_entropy"
```

The motivation is to see the advantage of the larger kernel size, as well as the inverted bottleneck layer. A more detailed discussion is provided later on.

| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| convnext_base | experiments/training_configs/classification/convnext_base.yaml | | | | |
| convnext_large | experiments/training_configs/classification/convnext_large.yaml | | | | |


## Generation

### The standard generation task:

For this task we use on the fly dataset generation with noise. Refer the dataset config in the earlier section for details.

`CTCLoss` is used for this task.

The code base currently supports three types of sequence models: `rnn`, `bilstm` and `transformer_encoder` for the `standard_generation` pipeline. The goal is to pose this as a non-autoregressive generation task, where a convolution based image encoder provides feature maps which is then reshaped and passed to the sequence model. While this is traditional, operations like attention could provide advantages when utilized in the primary image encoding as well, and recent work like SATRN and SVTR use attention in the primary image encoding.

#### Baseline: RNN vs BiLSTM vs Transformer

The configs for each are provided here for reference:

rnn:
```yaml
sequence_model_type: 'rnn'
sequence_model_config:
    hidden_size: 256
    num_layers: 2
    dropout: 0.1
    bidirectional: False
```

bilstm:
```yaml
sequence_model_type: 'bilstm'
sequence_model_config:
    hidden_size: 256
    num_layers: 2
    dropout: 0.1
```

transformer_encoder:
```yaml
sequence_model_type: 'transformer_encoder'
sequence_model_config:
    n_layers: 4
    d_model: 256
    n_heads: 8
    d_mlp: 1024
    n_ctx: 128
    d_vocab: 62
    act_fn: 'gelu'
    # attention is bi-directional
    # RoPE is used for position encoding
```
| Experiment Name | Config Path | Wandb Link| Kaggle Notebook Link | Kaggle Notebook Version | Results |
| --- | --- | --- | --- | --- | --- |
| rnn | experiments/training_configs/generation/rnn.yaml | | | | |
| bilstm | experiments/training_configs/generation/bilstm.yaml | | | | |
| transformer_encoder | experiments/training_configs/generation/transformer_encoder.yaml | | | | |


#### Alternate architectures:

At this point, it is important to take a deeper look at the task at hand, and see if there are modifications that could be done to the baseline models to improve performance.

The clean dataset is similar in spirit to an OCR task, the main deviations are:
- Different fonts and capitalizations occur within the same image
- The words do not themselves have any meaning/ distribution associated with it (e.g., they are not English words)

The best performing OCR models tend to be large VLMs, and largely this seems to be attributed to the language capabilities of the decoder. (\cite: relevant papers). Most of these are ViT based, and seem to be challenging to train at a sufficient scale for this task. 

The noisy dataset has further deviations:
- Noisy backgrounds 
- Character warping and rotations
- Character offsets and spacing
- Noisy strokes and dots (making relying on only local features potentially misleading)


### The 'difficult' generation task:
Will be added later.