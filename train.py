#!/usr/bin/env python3

"""Trains CLOOB."""

import argparse
import json
import pickle
import pprint

import clip
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import trange, tqdm
import webdataset as wds

from cloob_training.loss import cloob_loss
from cloob_training.model import get_and_init_model, set_precision


def psplit(x, n):
    return jax.tree_map(lambda x: jnp.stack(jnp.split(x, n)), x)


def unreplicate(x):
    return jax.tree_map(lambda x: x[0], x)


class TokenizerWrapper:
    def __init__(self, context_length=77):
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.context_length = context_length
        self.max_len = self.context_length - 2

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros([len(texts), self.context_length], dtype=torch.long)
        for i, text in enumerate(texts):
            tokens_trunc = self.tokenizer.encode(text)[:self.max_len]
            tokens = [self.sot_token, *tokens_trunc, self.eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result


class RandomItem:
    def __call__(self, batch):
        index = torch.randint(len(batch), [])
        return batch[index]    


def inverse_decay_schedule(init_value, steps=1., power=1., warmup=0., final_lr=0.):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup.

    steps is the number of steps required for the learning rate to decay to
    (1 / 2)**power of its original value.

    Args:
        init_value (float): The initial learning rate.
        steps (float): Inverse multiplicative factor of learning rate decay.
            Default: 1.
        power (float): Exponential factor of learning rate decay.
            Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate.
            Default: 0.
    """
    if not 0. <= warmup < 1:
        raise ValueError('Invalid value for warmup')
 
    def schedule(count):
        return init_value * (1 - warmup ** count) * jnp.maximum(final_lr, (1 + count / steps) ** -power)

    return schedule


def cosine_decay_schedule(init_value, steps, warmup=0., final_lr=0.):
    """Implements an cosine learning rate schedule with an optional exponential
    warmup.
        init_value (float): The initial learning rate.
        steps (float): The number of steps to reach the final learning rate.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate.
            Default: 0.
    """
    if not 0. <= warmup < 1:
        raise ValueError('Invalid value for warmup')

    def schedule(count):
        decay_range = init_value - final_lr
        lr_pre_warmup = decay_range * jnp.cos(jnp.minimum(count / steps, 1.) * jnp.pi / 2) ** 2 + final_lr
        return lr_pre_warmup * (1 - warmup ** count)

    return schedule


def make_weight_decay_mask(params):
    mask = {}
    for layer_name in params:
        mask[layer_name] = {}
        for param_name in params[layer_name]:
            mask[layer_name][param_name] = param_name == 'w' and not layer_name.endswith('embed')
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_config', type=str,
                   help='the model config file')
    p.add_argument('training_config', type=str,
                   help='the training config file')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    args = p.parse_args()

    config = json.load(open(args.model_config))
    training_config = json.load(open(args.training_config))
    dataset_config = training_config['dataset']
    opt_config = training_config['optimizer']
    sched_config = opt_config['schedule']
    wandb_config = training_config['wandb']
    print('Model config:')
    pprint.pprint(config)
    print('\nTraining config:')
    pprint.pprint(training_config)
    print()

    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_processes = jax.process_count()
    local_rank = jax.process_index()
    batch_size = training_config['batch_size_per_device'] * num_local_devices
    distributed_config = {
        'num_devices': num_devices,
        'num_local_devices': num_local_devices,
        'world_size': num_processes,
        'local_rank': local_rank,
        'batch_size_per_device': training_config['batch_size_per_device'],
        'local_batch_size': batch_size,
        'batch_size': batch_size * num_processes,
    }
    print('\nDistributed config:')
    pprint.pprint(distributed_config)
    print()

    image_size = config['image_encoder']['image_size']
    text_size = config['text_encoder']['text_size']
    set_precision(training_config['precision'])

    assert sched_config['type'] in ('inverse', 'cosine')
    if sched_config['type'] == 'inverse':
        schedule = inverse_decay_schedule(sched_config['lr'],
                                          steps=sched_config['steps'],
                                          power=sched_config['power'],
                                          warmup=sched_config['warmup'],
                                          final_lr=sched_config['final_lr'])
    elif sched_config['type'] == 'cosine':
        schedule = cosine_decay_schedule(sched_config['lr'],
                                         steps=sched_config['steps'],
                                         warmup=sched_config['warmup'],
                                         final_lr=sched_config['final_lr'])

    key = jax.random.PRNGKey(training_config['seed'])
    key, subkey = jax.random.split(key)
    params, image_apply, text_apply = get_and_init_model(config, subkey)
    if not args.resume:
        epoch = 0
        opt_state = None
    else:
        print('Loading checkpoint:', args.resume)
        ckpt = pickle.load(open(args.resume, 'rb'))
        epoch = ckpt['epoch']
        params = jax.tree_map(jnp.array, ckpt['params'])
        opt_state = jax.tree_map(jnp.array, ckpt['opt_state'])
        del ckpt

    assert opt_config['type'] == 'adamw'
    weight_decay_mask = tuple(make_weight_decay_mask(submodel) for submodel in params)
    opt = optax.adamw(schedule,
                      b1=opt_config['beta_1'],
                      b2=opt_config['beta_2'],
                      eps=opt_config['eps'],
                      weight_decay=opt_config['weight_decay'],
                      mask=weight_decay_mask)  
    if opt_state is None:
        opt_state = opt.init(params)

    model_stats = {
        'image_params': hk.data_structures.tree_size(params[0]),
        'text_params': hk.data_structures.tree_size(params[1]),
        'total_params': hk.data_structures.tree_size(params),
    }

    print('Image encoder parameters:', model_stats['image_params'])
    print('Text encoder parameters:', model_stats['text_params'])
    print('Total parameters:', model_stats['total_params'])
    print()

    params = jax.device_put_replicated(params, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image_encoder']['normalize']['mean'],
                             std=config['image_encoder']['normalize']['std'])
    ])

    assert config['text_encoder']['tokenizer'] == 'clip'
    ttf = transforms.Compose([
        TokenizerWrapper(text_size),
        RandomItem(),
    ])

    assert dataset_config['type'] == 'webdataset'
    dataset = wds.DataPipeline(
        wds.ResampledShards(dataset_config['location']),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1000, handler=wds.warn_and_continue),
        wds.decode('pilrgb', handler=wds.warn_and_continue),
        wds.to_tuple('jpg', 'txt', handler=wds.warn_and_continue),
        wds.map_tuple(tf, ttf, handler=wds.warn_and_continue),
    ).with_epoch(dataset_config['epoch_size'] // (num_processes * dataset_config['num_workers']))
    loader = data.DataLoader(dataset, batch_size, drop_last=True,
                             num_workers=dataset_config['num_workers'],
                             persistent_workers=True)

    def loss_fn(params, inputs, axis_name='i'):
        image_features = image_apply(params[0], inputs[0])
        text_features = text_apply(params[1], inputs[1])
        image_features = jax.lax.all_gather(image_features, axis_name).reshape([-1, image_features.shape[-1]])
        text_features = jax.lax.all_gather(text_features, axis_name).reshape([-1, text_features.shape[-1]])
        return cloob_loss(image_features, text_features, config['inv_tau'], config['scale_hopfield'])

    def train_step(params, opt_state, inputs, axis_name='i'):
        loss_grads = jax.value_and_grad(loss_fn)(params, inputs, axis_name)
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    pmap_train_step = jax.pmap(train_step, axis_name='i')

    def train_one_epoch(params, opt_state):
        for i, batch in enumerate(tqdm(loader)):
            inputs = jax.tree_map(lambda x: psplit(jnp.array(x), num_local_devices), batch)
            loss, params, opt_state = pmap_train_step(params, opt_state, inputs)
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
            if wandb_config['use_wandb'] and local_rank == 0:
                log_dict = {
                    'loss': unreplicate(loss).item(),
                    'epoch': epoch,
                }
                wandb.log(log_dict)
        return params, opt_state

    def save():
        obj = {'params': unreplicate(params),
                'opt_state': unreplicate(opt_state),
                'epoch': epoch}
        with open(training_config['checkpoint_name'], 'wb') as f:
            pickle.dump(obj, f)

    if wandb_config['use_wandb'] and local_rank == 0:
        import wandb
        wandb.init(project=wandb_config['project'],
                   entity=wandb_config['entity'],
                   group=wandb_config['group'],
                   config={'model': config,
                           'training': training_config,
                           'distributed': distributed_config,
                           'model_stats': model_stats},
                   save_code=True)

    try:
        while True:
            tqdm.write(f'Epoch {epoch}')
            params, opt_state = train_one_epoch(params, opt_state)
            epoch += 1
            tqdm.write('')
            save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
