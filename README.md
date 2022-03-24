# cloob-training

## Pretrained models

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
