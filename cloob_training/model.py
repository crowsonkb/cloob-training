import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class PositionalEmbedding(hk.Module):
    def __init__(self, init=None, name=None):
        super().__init__(name=name)
        self.init = hk.initializers.RandomNormal(1, 0) if init is None else init

    def __call__(self, x):
        w = hk.get_parameter('w', x.shape[1:], init=self.init)
        return x + w


class SelfAttention(hk.Module):
    def __init__(self, num_heads=1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads

    def __call__(self, x, padding_mask=None):
        d_model = x.shape[-1]
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        padding_mask = None if padding_mask is None else padding_mask[:, None, :, None]    # This is wrong! It should be [:, None, None, :].
        x = hk.MultiHeadAttention(self.num_heads, x.shape[-1] // self.num_heads, 1.)(x, x, x, padding_mask)
        return x


class FeedForward(hk.Module):
    def __init__(self, d_ff, name=None):
        super().__init__(name=name)
        self.d_ff = d_ff

    def __call__(self, x):
        d_model = x.shape[-1]
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = hk.Linear(self.d_ff, name='linear_0')(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(d_model, name='linear_1')(x)
        return x


class TransformerEncoderLayer(hk.Module):
    def __init__(self, d_ff, n_heads, name=None):
        super().__init__(name=name)
        self.d_ff = d_ff
        self.n_heads = n_heads
        
    def __call__(self, x, padding_mask=None):
        x = x + SelfAttention(self.n_heads)(x, padding_mask)
        x = x + FeedForward(self.d_ff)(x)
        return x


class TextEncoder(hk.Module):
    def __init__(self, d_embed, n_layers, d_model, n_heads, vocab_size, name=None):
        super().__init__(name=name)
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = self.d_model * 4
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.eot_token = vocab_size - 1

    def __call__(self, x):
        eot_mask = x == self.eot_token
        padding_mask = jnp.cumsum(eot_mask, axis=-1) == 0 | eot_mask
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(self.d_model))
        x = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.d_embed, w_init=w_init, name='embed')(x)
        x = PositionalEmbedding(w_init, name='pos_embed')(x)
        for i in range(self.n_layers):
            x = TransformerEncoderLayer(self.d_ff, self.n_heads, name=f'layer_{i}')(x, padding_mask)
        x = x[:, 0]
        x = hk.Linear(self.d_embed, name='proj')(x)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x


class ViTImageEncoder(hk.Module):
    def __init__(self, d_embed, n_layers, d_model, n_heads, patch_size, name=None):
        super().__init__(name=name)
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = self.d_model * 4
        self.n_heads = n_heads
        self.patch_size = patch_size

    def __call__(self, x):
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(self.d_model))
        x = hk.Conv2D(self.d_model, self.patch_size, self.patch_size, with_bias=False, w_init=w_init, data_format='NCHW', name='embed')(x)
        x = x.reshape([x.shape[0], x.shape[1], -1]).transpose([0, 2, 1])
        class_embed = hk.get_parameter('class_embed', shape=[self.d_model], init=w_init)
        class_embed = jnp.tile(class_embed[None, None], (x.shape[0], 1, 1))
        x = jnp.concatenate([class_embed, x], axis=1)
        x = PositionalEmbedding(w_init, name='pos_embed')(x)
        for i in range(self.n_layers):
            x = TransformerEncoderLayer(self.d_ff, self.n_heads, name=f'layer_{i}')(x)
        x = x[:, 0]
        x = hk.Linear(self.d_embed, name='proj')(x)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x


def get_hk_model(config):
    assert config['image_encoder']['type'] == 'ViT'
    assert config['text_encoder']['type'] == 'transformer'
    image_model_fn = lambda *args: ViTImageEncoder(
        config['d_embed'],
        config['image_encoder']['n_layers'],
        config['image_encoder']['d_model'],
        config['image_encoder']['n_heads'],
        config['image_encoder']['patch_size'],
    )(*args)
    text_model_fn = lambda *args: TextEncoder(
        config['d_embed'],
        config['text_encoder']['n_layers'],
        config['text_encoder']['d_model'],
        config['text_encoder']['n_heads'],
        config['text_encoder']['vocab_size'],
    )(*args)
    image_model = hk.without_apply_rng(hk.transform(image_model_fn))
    text_model = hk.without_apply_rng(hk.transform(text_model_fn))
    return image_model, text_model


def get_and_init_model(config, key):
    image_model, text_model = get_hk_model(config)
    key, subkey = jax.random.split(key)
    image_size = config['image_encoder']['image_size']
    input_channels = config['image_encoder']['input_channels']
    image_params = image_model.init(key,
                                    jnp.zeros([1, input_channels, image_size, image_size]))
    text_size = config['text_encoder']['text_size']
    text_params = text_model.init(subkey,
                                  jnp.zeros([1, text_size], dtype=jnp.int32))
    return (image_params, text_params), image_model.apply, text_model.apply


def set_precision(mode):
    import jmp
    if mode == 'float32':
        policy = jmp.get_policy('float32')
    elif mode == 'float16':
        policy = jmp.get_policy('params=float32,compute=float16,output=float32')
    elif mode == 'bfloat16':
        policy = jmp.get_policy('params=float32,compute=bfloat16,output=float32')
    else:
        raise ValueError('mode should be one of float32, float16, bfloat16')
    hk.mixed_precision.set_policy(TextEncoder, policy)
    hk.mixed_precision.set_policy(ViTImageEncoder, policy)
