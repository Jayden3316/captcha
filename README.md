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
Width is then rescaled to the nearest value to 28k + 14. Input images are RGB (3 channels).

### The Encoder

#### Architecture 1:

Convolution layers (filter size, stride, num_channels) are as follows:

- Conv2D 1: [7, 1, 16]     (Height is now 64)
- Activation (SiLU/Swish)
- Max pooling (stride=2)   (Height is now 32)
- Conv2D 2: [5, 1, 32]     (Height is now 28)
- Activation (SiLU/Swish)
- Max pooling (stride=2)   (Height is now 14)

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

The motivation for the rather aggressive downsampling used is to see whether we could have a rather small context window - further, this also helps avoid having multiple rows of patches. Intuitively while transformers are capable of identifying such patterns, you would need significant depth to achieve recognizing certain features. (E.g., to identify a line that runs vertically down the page, you would need to realize that similar tokens appear every k steps, and k is variable since this model supports variable widths.) The goal was to force such behaviour through inductive biases that convolution naturally offers.

#### Architecture 2:
Previously the entire model had 1.7 M parameters, and Linear 1 contributed to 1.2 M of those parameters, which seems sub-optimal. The convolution blocks contribute 15k parameters despite the aggressive compression applied. Finally, the smallest dataset used (based on wiki-10k) has images between 70 x 42 to nearly 70 x 1610 images, a vocabulary size of 63 and 22k+ unique words over 19 font families, making it reasonably challening in itself heuristically. 

Based on these observations, the goal is to increase the amount of feature extraction possible in the vision encoder. The proposed method is to add ConvNext blocks between each of the existing convolution layers.

##### The ConvNext Block:

Each block is as follows:
```
class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value >0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) [original ConvNext implements this for speed improvements]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x
```
Each ConvNextBlock adds 10k params if num_input channels is set to 32. The updated encoder is as follows:

Other changes include using nn.GELU() everywhere as well as using depthwise convolutions everywhere
```
class CNNEncoder2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=1)
        
        self.cnxt1 = nn.Sequential(
            ConvNextBlock(dim=16),
            ConvNextBlock(dim=16),
            ConvNextBlock(dim=16)
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=1)
        
        self.cnxt2 = nn.Sequential(
            ConvNextBlock(dim=32),
            ConvNextBlock(dim=32),
            ConvNextBlock(dim=32)
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.GELU()

    def forward(self, x):
      x = self.act(self.conv1(x)) # [b, 16, 64, w-6]
      x = self.cnxt1(x)
      x = self.pool1(x) # [b, 16, 32, (w-6)//2]

      x = self.act(self.conv2(x)) # [b, 32, 28, ...]
      x = self.cnxt2(x)
      x = self.pool2(x) # [b, 32, 14, w_final]

      patches = x.unfold(3, 7, 7)
      patches = patches.permute(0, 3, 1, 2, 4).continguous()
      b, num_tokens, c, h, w_patch = patches.shape
      out = patches.view(b, num_tokens, c * h * w_patch)

      return out
```
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