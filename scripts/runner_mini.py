import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path("train_mnist.py")
OUT_ROOT = Path("..") / "artifacts"


def run_mnist(
        *,
        seed: int = 1,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 3,
        device: str = "gpu",
        track: bool = False,
) -> None:
    """
    Launch a single MNIST training run as a subprocess.
    """

    run_name = f"mnist__{optimizer}__seed{seed}__lr{learning_rate}"

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-name", run_name,
        "--out-root", str(OUT_ROOT),
        "--seed", str(seed),
        "--optimizer", optimizer,
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--max-epochs", str(max_epochs),
        "--device", device,
        "--track", str(track),
    ]

    if optimizer == "sgd":
        cmd += ["--momentum", "0.9"]

    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_mnist()
