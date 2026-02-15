import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_LOG_COMPILES", "0")

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import tensorflow_datasets as tfds
import tensorflow as tf

import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels
from optax import adam, sgd
from flax import linen as nn


# -----------------------------
# Args
# -----------------------------

def bool_flag(x: str) -> bool:
    x = x.strip().lower()
    if x in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {x}")


def parse_args():
    p = argparse.ArgumentParser()

    # ====== Data / Experiment ======
    p.add_argument("--dataset", type=str, default="mnist", help="TFDS dataset id (mnist expected)")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=-1, help="Overrides epochs if > 0")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)

    # ====== Model ======
    p.add_argument("--model", type=str, default="cnn-m", choices=["cnn-s", "cnn-m"])

    # ====== Optimization ======
    p.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")

    # ====== Hardware ======
    p.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])

    # ====== Tracking ======
    p.add_argument("--track", type=bool_flag, default=True, nargs="?", const=True)
    p.add_argument("--wandb-project-name", type=str, default="mle4r")
    p.add_argument("--wandb-entity", type=str, default="dysco")

    # ====== Outputs ======
    p.add_argument("--out-root", type=str, default=str(Path("..") / "artifacts"))
    p.add_argument("--run-name", type=str, default=None, help="If None, uses dataset__opt__timestamp")

    return p.parse_args()


# -----------------------------
# Data (in-memory)
# -----------------------------

def load_mnist_in_memory(dataset_id: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads full train/test into host RAM (NumPy).
    For MNIST this is tiny and simplifies the script.
    """
    # Prevent TF from taking the GPU
    tf.config.set_visible_devices([], device_type="GPU")

    # TFDS returns dicts unless as_supervised=True
    (train_ds, test_ds), _info = tfds.load(
        dataset_id,
        data_dir=str(data_dir),
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )

    def to_np(ds):
        xs, ys = [], []
        for x, y in tfds.as_numpy(ds):
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)
        Y = np.asarray(ys, dtype=np.int32)
        return X, Y

    X_tr, y_tr = to_np(train_ds)
    X_te, y_te = to_np(test_ds)

    # Cast once
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    return X_tr, y_tr, X_te, y_te


# -----------------------------
# Model
# -----------------------------

def batch_agnostic_reshape(x, x_dims=4):
    # x_dims includes batch for conv outputs; we just flatten per-example
    if len(x.shape) == x_dims:
        return x.reshape((x.shape[0], -1))
    return x.reshape((x.shape[0], -1))


class CNNClassifierSmall(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        x = x / 255.0
        x = nn.Conv(16, (3, 3), (1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(32, (3, 3), (1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = batch_agnostic_reshape(x, x_dims=4)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


class CNNClassifierMedium(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        x = x / 255.0
        x = nn.Conv(32, (3, 3), (1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(64, (3, 3), (1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = batch_agnostic_reshape(x, x_dims=4)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("train_mnist")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Outputs / run identity
    out_root = Path(args.out_root)
    tb_root = out_root / "tensorboard"
    data_root = out_root / "data"

    run_name = args.run_name or f"{args.dataset}__{args.optimizer}__{int(time.time())}"
    run_dir = out_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = vars(args).copy()
    config.update(
        jax_device=str(jax.devices()[0]),
        jax_platform=jax.devices()[0].platform,
        run_name=run_name,
        run_dir=str(run_dir),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    log.info(f"run_name: {run_name}")
    log.info(f"run_dir: {run_dir}")
    log.info(f"JAX device: {config['jax_device']}")

    # Optional W&B
    wandb_run = None
    if args.track:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=config,
                name=run_name,
                save_code=True,
            )
        except Exception as e:
            log.warning(f"W&B disabled (init failed): {e}")
            wandb_run = None

    writer = SummaryWriter(str(tb_root / run_name))

    # Data in memory
    X_tr_np, y_tr_np, X_te_np, y_te_np = load_mnist_in_memory(args.dataset, data_dir=data_root)
    n_train = X_tr_np.shape[0]
    n_classes = int(np.max(y_tr_np)) + 1
    log.info(f"Loaded: train={X_tr_np.shape}, test={X_te_np.shape}, classes={n_classes}")

    # JAX arrays for test (kept on device)
    X_test = jnp.asarray(X_te_np, dtype=jnp.float32)
    y_test = jnp.asarray(y_te_np, dtype=jnp.int32)

    # Model
    model = CNNClassifierSmall(n_classes) if args.model == "cnn-s" else CNNClassifierMedium(n_classes)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    dummy_x = jnp.ones((args.batch_size, 28, 28, 1), dtype=jnp.float32)
    params = model.init(init_key, dummy_x)["params"]

    # Optimizer
    if args.optimizer == "sgd":
        tx = sgd(args.learning_rate, momentum=args.momentum)
    else:
        tx = adam(args.learning_rate)
    opt_state = tx.init(params)

    def predict_fn(p, xs):
        return model.apply({"params": p}, xs)

    def ce(p, x, y):
        logits = predict_fn(p, x)
        return jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y))

    @jax.jit
    def accuracy(p, x, y):
        logits = predict_fn(p, x)
        pred = jnp.argmax(logits, axis=1)
        return jnp.mean(pred == y)

    @jax.jit
    def train_step(p, s, xs, ys):
        loss, grads = jax.value_and_grad(ce)(p, xs, ys)
        updates, s2 = tx.update(grads, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, loss

    # Warmup compile
    warm_x = dummy_x
    warm_y = jnp.zeros((args.batch_size,), dtype=jnp.int32)
    p2, s2, l2 = train_step(params, opt_state, warm_x, warm_y)
    jax.block_until_ready(l2)
    params, opt_state = p2, s2

    # Steps
    steps_per_epoch = n_train // args.batch_size
    steps_from_epochs = args.max_epochs * steps_per_epoch
    steps_from_max = args.max_steps if args.max_steps > 0 else 10 ** 18
    total_steps = int(min(steps_from_epochs, steps_from_max))

    eval_every = args.eval_every
    log_every = args.log_every

    # Initial eval
    acc0 = float(jax.device_get(accuracy(params, X_test, y_test)))
    writer.add_scalar("eval/accuracy", acc0, 0)
    writer.add_scalar("eval/wall_clock_train", 0.0, 0)
    writer.add_scalar("eval/wall_clock_total", 0.0, 0)

    # Timing
    t_total0 = time.time()
    t_train_window0 = time.time()
    train_time_accum = 0.0

    # Batch sampler
    rng_np = np.random.default_rng(args.seed)

    def sample_batch():
        idx = rng_np.integers(0, n_train, size=(args.batch_size,), endpoint=False)
        xb = jnp.asarray(X_tr_np[idx], dtype=jnp.float32)
        yb = jnp.asarray(y_tr_np[idx], dtype=jnp.int32)
        return xb, yb

    for step in tqdm(range(1, total_steps + 1)):
        batch_x, batch_y = sample_batch()
        params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)

        if step % log_every == 0:
            jax.block_until_ready(loss)
            t_now = time.time()
            dt_train = t_now - t_train_window0
            train_time_accum += dt_train
            sps = log_every / max(dt_train, 1e-12)

            writer.add_scalar("train/loss", float(jax.device_get(loss)), step)
            writer.add_scalar("speed/sps", sps, step)

            t_train_window0 = time.time()

        if step % eval_every == 0:
            acc = float(jax.device_get(accuracy(params, X_test, y_test)))
            wall_total = time.time() - t_total0

            writer.add_scalar("eval/accuracy", acc, step)
            writer.add_scalar("eval/wall_clock_train", train_time_accum, step)
            writer.add_scalar("eval/wall_clock_total", wall_total, step)

    writer.close()

    # Final metrics summary
    final_acc = float(jax.device_get(accuracy(params, X_test, y_test)))
    wall_total = time.time() - t_total0

    metrics = {
        "final_test_acc": final_acc,
        "total_steps": total_steps,
        "wall_clock_total": wall_total,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    log.info(f"Wrote metrics to {metrics_path}")

    # W&B bookeeping
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    log.info("Done.")


if __name__ == "__main__":
    main()
