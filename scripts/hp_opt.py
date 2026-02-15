import json
import numpy as np
from pathlib import Path

from runner_mini import run_mnist

OUT_ROOT = Path("..") / "artifacts"


def sample_config(rng: np.random.Generator):
    optimizer = rng.choice(["adam", "sgd"])
    learning_rate = 10 ** rng.uniform(-4, -1)  # log-uniform
    batch_size = rng.choice([64, 128])

    cfg = {
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
    }

    if optimizer == "sgd":
        cfg["momentum"] = float(rng.uniform(0.0, 0.95))

    return cfg


def read_metrics(run_name: str):
    path = OUT_ROOT / "runs" / run_name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    rng = np.random.default_rng(0)

    n_trials = 8
    max_epochs = 2  # small budget

    results = []

    for t in range(n_trials):
        cfg = sample_config(rng)

        print(f"\n=== Trial {t} ===")
        print(cfg)

        run_mnist(
            seed=t,
            optimizer=cfg["optimizer"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            max_epochs=max_epochs,
            track=False,
        )

        metrics = read_metrics(f"mnist__{cfg['optimizer']}__seed{t}__lr{cfg['learning_rate']}")
        if metrics is None:
            continue

        score = metrics.get("final_test_acc", 0.0)
        results.append((score, cfg))

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n==== BEST CONFIG ====")
    print("Score:", results[0][0])
    print("Config:", results[0][1])


if __name__ == "__main__":
    main()
