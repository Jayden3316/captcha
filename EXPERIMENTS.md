# Experiments for Captcha OCR

Tracks all experiments performed with paths to their configs, wandb logs, kaggle notebook version number for runs and results.

Please visit: https://www.kaggle.com/code/j11045/captcha for runs, model checkpoints and logs. 
Checkpoints are available at: `/experiments/<task>/<experiment_name>/checkpoints`

Training config has been reproduced here for reference. On the fly dataset generation is used for all training runs. Number of parameters is set to be of the order of the parameters of the model, however, in order to maximize the number of experiments that could be run within the compute budget on kaggle, all runs were terminated once target metric (validation accuracy for classification and exact match for generation) seemed to have reached a plateau. On average, classification tasks took 30 minutes on P100 and generation tasks took 1 hour on P100. All experiments cumulatively have used 19 hours of compute.

Hyperparameter sweep was not done for any experiment. The hyperparameters below were shared for all experiments.

```yaml
training_config:
  batch_size: 32
  
  training_steps: 50000 # num_samples = training_steps * batch_size = 1.6 * 10^6 
  # training_steps is set to be of the order of the number of parameters in the model
  # should be adjusted for every model
  # for generation tasks, the number of possible words are huge
  # so the assumption is that the problem is model constrained.

  use_onthefly_generation: True
  save_every_steps: 512
  val_check_interval_steps: 512
  val_steps: 128

  learning_rate: 0.0001
  optimizer_type: "adamw"
  weight_decay: 0.01
  grad_clip_norm: 1.0
  accumulate_grad_batches: 1
  
  checkpoint_dir: "experiments/<task>/<experiment_name>/checkpoints" # updated as needed


  save_every_n_epochs: 1
  monitor_metric: "exact_match" # for generation; 'val_acc' for classification

  wandb_project: "captcha-ocr"
  # wandb_name: "<experiment_name>" 
  log_every_n_steps: 10
  
  device: "cuda"
  num_workers: 4 # works well on kaggle. On colab, change as needed. 
  # On testing, colab seemed to be cpu bottlenecked
  # used for generating on the fly dataset

  mixed_precision: false
  shuffle_train: true

  metrics: ['character_accuracy', 'word_correct', 'edit_distance']
```
## Datasets

The config for noisy dataset generation is provided here for reference:

`ctr_p` (for resnets) | `noisy` (for convnext) | `chr_transform` (for classification tasks):

```yaml
dataset_config:
    width: 192
    height: 64
    width_divisor: 32
    width_bias: 0
    resize_mode: "variable"
    image_ext: "png"

    train_font_root: "./train_font_library" # contains 20 fonts
    val_font_root: "./val_font_library" # contains 5 held out fonts not in training set
    
    fonts: []
    max_fonts_per_family: 1 # some fonts have variants, but we use only one variant per font
    font_sizes: [42]

    #word_path: "experiments/diverse_words.tsv" # uncomment for classification tasks

    # diverse_words.tsv: 100 alphanumeric words of length 5
    # diverse_words_variable.tsv: 100 alphanumeric words of variable length between 4 and 10. Each length appears at least 10 times

    random_capitalize: False # since alphanumeric characters already contain mixed capitalization
    add_noise_dots: True
    add_noise_curve: True

    noise_bg_density: 0

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

- For `clean`: `add_noise_dots: False`, `add_noise_curve: False`, other parameters are set to 0
- For `noise_dots`: `add_noise_dots: True`, `add_noise_curve: True`, other parameters are set to 0

The height of all images is resized to 64x. There are two configurations for width handling - either resizing to a fixed width or resizing to the nearest width_divisor multiple.
- `fixed`: `resize_mode: "fixed"`
- `variable`: `resize_mode: "variable"`, `width_divisor: 32`


## Classification

### Comparing architectures

There are primarily two kinds of experiments in this subsection:
- Comparing between ConvNext and ResNet
- Setting a baseline performance of using a `SequenceModel` after the image encoder.

#### Architectural considerations:

Since captchas are of variable width, handling them seem to have a few variations:
- Resizing to a fixed width
- Pooling the feature map along the spatial dimensions
- Using a sequence model to process the feature map
- Use models trained on the generative task, then choose the label with least edit distance (and for better performance, use least edit distance on lowercase of predicted and true labels)

Pooling would result in a loss of spatial information, but this could be handled if there are a sufficient number of channels.

Resizing to a fixed width would harm longer captchas and compressing these might not be optimal, however, on experimentations, there did not seem to be a significant difference in performance upto captchas with 10 characters.

There are previous work that use a sequence model after an encoder like CRNN. Here, the feature maps from the ImageEncoder are reshaped and passed as visual tokens. The sequence model then processes these visual tokens to generate the final output. Discussion on such models is deferred to the `generation` section.

By default, both image encoders use the following pattern:


| Block | `Input_dims` -> `Output_dims` | Description |
| --- | --- | --- |
| Stem:          |    `[B, C, H, W] -> [B, C', H/4, W/4]`               |   4x downsample               |
| Block1:        |    `[B, C', H/4, W/4] -> [B, C', H/4, W/4]`          |   consists of k blocks        |
| Downsample1:   |    `[B, C', H/4, W/4] -> [B, C'', H/8, W/8]`         |   2x downsample               |
| Block2:        |    `[B, C'', H/8, W/8] -> [B, C'', H/8, W/8]`        |   consists of k blocks        |
| Downsample2:   |    `[B, C'', H/8, W/8] -> [B, C''', H/16, W/16]`     |   2x downsample               |
| Block3:        |    `[B, C''', H/16, W/16] -> [B, C''', H/16, W/16]`  |   consists of 3k blocks       |
| Downsample3:   |    `[B, C''', H/16, W/16] -> [B, C'''' H/32, W/32]`  |   2x downsample               |
| Block4:        |    `[B, C'''' H/32, W/32] -> [B, C'''' H/32, W/32]`  |   consists of k blocks        |

This means that each vision token covers a stride of at least 32px along the width and the entire height. Since the width of the characters are also approximately 20-40px, each token could cover multiple characters, especially if 'narrow' letters appear together. For the classification task however, this did not significantly affect performance. Further discussion on this is deferred to the `generation` section.

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
`resnet-medium` uses the same config as `resnet-base` but sets dims as `[16, 32, 48, 96]`.

convnext-base: 

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
        stage_block_counts: [3, 3, 9, 3]
        downsample_strides: [[2, 2], [2, 2], [2, 2]]
        downsample_kernels: [[2, 2], [2, 2], [2, 2]]
        downsample_padding: [[0, 0], [0, 0], [0, 0]]

        # the following are defaults, but reproduced here for clarity:
        convnext_kernel_size: 7
        convnext_drop_path__rate: 0.0
        convnext_expansion_ratio: 4

    adapter_type: "flatten"
    adapter_config:
        output_dim: 1536 # 128 * 2 * 6

    projector_type: 'linear'
    projector_config:
        output_dim: 576
    sequence_model_type: null

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

## Results

| Name | val_acc | val_f1 | val_loss | step | batch_size | dims | encoder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resnet_base_classification_clean_fixed |  0.962  | 0.9565 | 0.1053 | 10000 | 32 | [8,16,24,48] | resnet |
| resnet_base_classification_clean_variable |  0.956 |  0.9499 | 0.1200 | 10000 | 32 | [8,16,24,48] | resnet |
| convnext_base_classification_variable |  0.956 |  0.9485 | 0.1638 | 28680 | 32 | [16,32,64,128] | convnext |
| convnext_base_classification |  0.931  |  0.9226 |  0.2707 | 10000 | 32 | [16,32,64,128] | convnext |
| convnext_base_noise_dots_classification* | 0.852  |  0.8482 | 0.5025 | **6830** | 32 | [16,32,64,128] | convnext |
| convnext_base_noisy_classification* |  0.77 |  0.7607 | 0.7930 | 10000 | 32 | [16,32,64,128] | convnext |
| convnext_base_chr_transforms_classification* | 0.70  |  0.7003 | 1.0362 | 10000 | 32 | [16,32,64,128] | convnext |

- (*) These runs were terminated early. Plots show that they would have likely converged to a better accuracy.

![classification scores](images/classification.png)
## Generation

The effective stride along the width is determined by the embedding dimension since `VerticalFeatureAdapter` does the following transform: `[B, C, H, W] -> [B, W//f, C * H * f], where f = (output_dim // C * H)`. 


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