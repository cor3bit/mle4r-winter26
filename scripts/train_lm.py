import argparse
import time
from distutils.util import strtobool
import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels
from flax import linen as nn

from tensorboardX import SummaryWriter
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
import tiktoken


def parse_args():
    parser = argparse.ArgumentParser()

    # ====== Experiment ======
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed of the experiment")
    parser.add_argument("--max-iters", type=int, default=50_000,
                        help="Number of training iterations")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Number of training iterations between two consecutive evaluations")

    # ====== Model ======
    parser.add_argument("--tokenizer", type=str, default='char',
                        help="Tokenizer to use: 'char' or 'bpe'")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Context window for the transformer model")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Rate for dropout in the transformer model")
    parser.add_argument("--n-layer", type=int, default=6,
                        help="Number of layers for the transformer model")
    parser.add_argument("--n-embd", type=int, default=256,
                        help="Size of the embedding for the transformer model")
    parser.add_argument("--n-head", type=int, default=8,
                        help="Number of heads for the transformer model")
    parser.add_argument("--head-size", type=int, default=32,
                        help="Size of the heads for the transformer model")

    # ====== Optimizer Parameters ======
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="Optimizer to use")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate passed to the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Learning rate passed to the optimizer")

    # ====== Tracking ======
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="mle4r",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="dysco",
                        help="the entity (team) of wandb's project")

    args = parser.parse_args()

    return args


@partial(jax.jit, static_argnums=(2, 3))
def get_batch(random_key, data, batch_size, block_size):
    ix = jax.random.randint(
        random_key, shape=(batch_size, 1), minval=0, maxval=len(data) - block_size
    )
    x = dynamic_slice_vmap(data, ix, (block_size,))
    y = dynamic_slice_vmap(data, ix + 1, (block_size,))
    return x, y


@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x, training=False)
    loss = ce_loss(logits=logits, labels=y)
    return loss


def ce_loss(logits, labels):
    return softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()


# optimization step
@partial(jax.jit, static_argnums=(3, 4))
def step(key, params, opt_state, batch_size, block_size):
    key, subkey = jax.random.split(key)
    batch = get_batch(key, train_data, batch_size, block_size)
    loss, grad = jax.value_and_grad(loss_fun)(params, *batch, subkey)
    updates, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, key, opt_state, loss


class NanoLM(nn.Module):
    vocab_size: int
    num_layers: int = 6
    num_heads: int = 8
    head_size: int = 32
    dropout_rate: float = 0.2
    embed_size: int = 256
    block_size: int = 64

    @nn.compact
    def __call__(self, x, training: bool = True):
        seq_len = x.shape[1]

        x = nn.Embed(self.vocab_size, self.embed_size)(x) + \
            nn.Embed(self.block_size, self.embed_size)(jnp.arange(seq_len))

        for _ in range(self.num_layers):
            x_norm = nn.LayerNorm()(x)

            x = x + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.head_size,
                out_features=self.head_size * self.num_heads,
                dropout_rate=self.dropout_rate,
            )(
                x_norm,
                x_norm,
                mask=jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]))),
                deterministic=not training,
            )

            x = x + nn.Sequential([
                nn.Dense(4 * self.embed_size),
                nn.relu,
                nn.Dropout(self.dropout_rate, deterministic=not training),
                nn.Dense(self.embed_size),
            ])(nn.LayerNorm()(x))

        x = nn.LayerNorm()(x)

        return nn.Dense(self.vocab_size)(x)

    @partial(jax.jit, static_argnames=("self", "length"))
    def generate(self, rng, params, length):
        def _scan_generate(carry, _):
            random_key, context = carry
            logits = self.apply(params, context, training=False)
            rng, rng_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(
                rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
            )
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (rng, context), new_token

        _, new_tokens = jax.lax.scan(
            _scan_generate,
            (rng, jnp.zeros((1, self.block_size), dtype=jnp.int32)),
            (),
            length=length,
        )
        return new_tokens


if __name__ == '__main__':
    def loss_fun(params, x, y, dropout_key):
        logits = model.apply(params, x, training=True, rngs={"dropout": dropout_key})
        loss = ce_loss(logits=logits, labels=y)
        return loss


    # --------------- START HERE ---------------
    # force jax to use CPU
    # jax.config.update('jax_platform_name', 'cpu')

    # for debugging JAX-related issues
    # jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)
    # jax.config.update('jax_disable_jit', True)

    # platform check
    print("JAX is running on", jax.devices()[0].platform.upper())

    # --------------- Arguments ---------------
    args = parse_args()
    run_name = f"nanolm_{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(os.path.join('..', 'artifacts', 'tensorboard', run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --------------- Data ---------------
    ds = tfds.load("tiny_shakespeare")

    # combine train and test examples into a single string
    text_train = ""
    for example in ds["train"].concatenate(ds["test"]).as_numpy_iterator():
        text_train += example["text"].decode("utf-8")

    # similarly, create a single string for validation
    text_validation = ""
    for example in ds["validation"].as_numpy_iterator():
        text_validation += example["text"].decode("utf-8")

    if args.tokenizer == "char":
        vocab = sorted(list(set(text_train)))
        vocab_size = len(vocab)

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for i, ch in enumerate(vocab)}
        encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers
        decode = lambda l: "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

        # transfer train and validation data to JAX arrays
        train_data = jnp.array(encode(text_train))
        eval_data = jnp.array(encode(text_validation))
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        vocab_size = enc.n_vocab

        encode = lambda s: enc.encode(s)
        decode = lambda l: enc.decode(l)

        # transfer train and validation data to JAX arrays
        train_data = jnp.array(enc.encode(text_train))
        eval_data = jnp.array(enc.encode(text_validation))

    # prepare a function for retrieving a batch of data
    dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))

    # --------------- Model ---------------
    model = NanoLM(
        vocab_size=vocab_size,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        head_size=args.head_size,
        dropout_rate=args.dropout,
        embed_size=args.n_embd,
        block_size=args.block_size,
    )

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    batch_size = args.batch_size
    block_size = args.block_size

    var_params = model.init(
        key,
        jnp.ones((batch_size, block_size), dtype=jnp.int32),
        training=False,
    )

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(var_params))

    print(f"Total number of parameters: {n_params:_}")

    if args.optimizer == "adamw":
        opt = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = optax.adam(learning_rate=args.learning_rate)
    elif args.optimizer == "sgd":
        opt = optax.sgd(learning_rate=args.learning_rate)
    else:
        raise ValueError("Unknown optimizer")

    opt_state = opt.init(var_params)

    # all_train_losses = []
    # all_eval_losses = []

    # ------------ Optimization Loop ------------
    n_iter = args.max_iters
    n_freq_eval = args.eval_interval

    for i in tqdm(range(n_iter)):
        var_params, key, opt_state, loss = step(key, var_params, opt_state, batch_size, block_size)
        # all_train_losses.append(loss)
        writer.add_scalar('train_loss', jax.device_get(loss), i)

        # once every N_FREQ_EVAL we compute loss on the validation set
        if i % n_freq_eval == 0:
            key, subkey = jax.random.split(key)
            eval_loss = eval_step(var_params, *get_batch(subkey, eval_data, batch_size, block_size))
            # all_eval_losses.append(eval_loss)
            print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_loss}")
            writer.add_scalar('test_loss', jax.device_get(eval_loss), i)

    # generate some text after training is done
    key, subkey = jax.random.split(key)
    text = model.generate(key, var_params, 1000)[:, 0, 0].tolist()
    print(decode(text))

    # bookkeeping
    writer.close()
    if args.track:
        wandb.finish()
