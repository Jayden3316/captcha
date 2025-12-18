# Fast Captcha + OCR

Implementing and experimenting with various models and architectures that can break captcha.

## Main functionality:
- Create a dataset of `.png` text based captcha images
  - Supports generation of captcha with various fonts (of your choice), variable captialization, noise strokes and variable background noise
- Train various architectures for breaking captcha with logging to wandb
- (Inspired by transformer_lens): Provide functionality to track activations and study where models could be improved
- Calculate relevant metrics like Levenshtein distance, exact matches, character accuracy etc.
- Test against other approaches like Tesseract or VLMs

## Architecture proposed

The architecture consists of an encoder-decoder structure. The encoder starts with a series of convolution blocks and then has 4 transformer layers.

The decoder has 4 transformer layers and uses a DETR-style object query mechanism for one-shot prediction (non-autoregressive).

Images are first rescaled to sizes of height 70 px and variable width.
Width is then rescaled to the nearest multiple of 35. Input images are RGB (3 channels).

### The Encoder

Convolution layers (filter size, stride, num_channels) are as follows:

- Conv2D 1: [7, 1, 16]     (Height is now 64)
- Activation (SiLU/Swish)
- Max pooling (stride=2)  (Height is now 32)
- Conv2D 2: [5, 1, 32]     (Height is now 28)
- Activation (SiLU/Swish)
- Max pooling (stride=2)  (Height is now 14)

Next these feature maps are pooled to give a tensor of dimension `(batch_size, token_height * token_width * num_channels, num_tokens)`.

Here token_height is 14 and token_width is 7. `num_tokens = int(width/7)`.

Then the following projections are applied:
At the end of the previous steps, the embedding dimension would be 3136 (32 * 14 * 7).
- Linear 1: 3136 -> 392
- Activation (SiLU)
- Linear 2: 392 -> 128
- Activation (SiLU)
- Linear 3: 128 -> 64

Now these matrices are passed to a transformer-encoder.

Each transformer block is of the following config:
- `embedding_dim = 64`
- first self attention with 4 attention heads
- MLP with `hidden_dim = 64 * 4` and activation as `GeGELU` (approximated by `GELU`)
- RMSNorm before self attention, cross attention and MLP
- RoPE in all layers

### The decoder

The decoder operates in a non-autoregressive manner (one-shot prediction).
The input to the decoder is a set of learnable object queries (parameters) of shape `(seq_len, embedding_dim)`, initialized from noise.

- `embedding_dim = 64`
- `seq_len = 56` (corresponds to the longest word that the model supports)

Each transformer block is of the following config:
- `embedding_dim = 64`
- first self attention with 4 attention heads
- cross attention with outputs of encoder with 4 attention heads (bidirectional context)
- MLP with `hidden_dim = 64 * 4` and activation as `GeGELU`
- RMSNorm before self attention, cross attention and MLP
- RoPE in all layers

Finally, layer_norm is applied on the residual stream and passed into the unembedding matrix. Softmax is applied to get character probabilities for each position independently. The vocabulary consists of letters and an additional `<PAD>` token.

## Usage

The project uses a unified CLI `main.py`.

### Generate Dataset
```
python 

captcha/main.py generate \
  --word-file path/to/words.tsv \
  --font-root path/to/fonts \
  --out-dir validation_set \
  --metadata-path validation_set/metadata.json
  
  
```
### Train Model

```
python 

captcha/main.py train \
  --metadata-path validation_set/metadata.json \
  --batch-size 32 \
  --epochs 10
  ```
  
## File structure
- `modelling.py`: Has the architecture (CNN Encoder + Transformer Encoder-Decoder) and forward pass 
- `config.py`: Implements the config class to specify encoder-decoder architecture, hyperparams, etc
- `train.py`: Implements the training and validation loops along with logging on wandb 
- `generate_captchas.py`: For generating the dataset 
- `utils.py`: Implements metrics (Levenshtein, Char Accuracy) and processor class for image resizing and tokenization.
- `main.py`: Entry point for generation and training commands.
- `test_tesseract_baseline.py`: Implements tesseract on the dataset of interest and provides metrics.