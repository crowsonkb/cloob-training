# cloob-training

## Pretrained models

There are two pretrained [CLOOB](https://ml-jku.github.io/cloob/) models in this repo at the moment, a 16 epoch and a 32 epoch ViT-B/16 checkpoint trained on [LAION 400M](https://laion.ai/laion-400-open-dataset/).

Zero-shot ImageNet validation set accuracy (using [OpenCLIP](https://github.com/mlfoundations/open_clip)'s code):

| Model name | Top 1 | Top 5 |
| ---------- | ----- | ----- |
| cloob_laion_400m_vit_b_16_16_epochs | 0.61238 | 0.8492  |
| cloob_laion_400m_vit_b_16_32_epochs | 0.62816 | 0.85964 |
| OpenAI CLIP ViT-B/32                | 0.6327  | 0.88772 |
| OpenAI CLIP ViT-B/16                | 0.68132 | 0.91768 |
| OpenAI CLIP ViT-L/14                | 0.75388 | 0.9454  |
| OpenAI CLIP RN50                    | 0.59806 | 0.86498 |
| OpenAI CLIP RN101                   | 0.62296 | 0.88106 |
| OpenAI CLIP RN50x4                  | 0.66268 | 0.9046  |
| OpenAI CLIP RN50x16                 | 0.70754 | 0.92822 |
| OpenAI CLIP RN50x64                 | 0.74134 | 0.94146 |


### PyTorch

```python
from cloob_training import model_pt, pretrained

pretrained.list_configs()
```

returns:

```
['cloob_laion_400m_vit_b_16_16_epochs', 'cloob_laion_400m_vit_b_16_32_epochs']
```

The models can be used by:

```python
config = pretrained.get_config('cloob_laion_400m_vit_b_16_16_epochs')
model = model_pt.get_pt_model(config)
checkpoint = pretrained.download_checkpoint(config)
model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
model.eval().requires_grad_(False).to('cuda')
```

Model class attributes:

`model.config`: the model config dict.

`model.image_encoder`: the image encoder, which expects NCHW batches of normalized images (preprocessed by `model.normalize`), where C = `model.config['image_encoder']['input_channels']` and H, W = `model.config['image_encoder']['image_size']`.

`model.text_encoder`: the text encoder, which expects text tokenized by `model.tokenize`.

`model.normalize`: the preprocessor for image tensors.

`model.tokenize`: the preprocessor for text.

### JAX

Coming soon...

## Training (JAX only)

Coming soon...
