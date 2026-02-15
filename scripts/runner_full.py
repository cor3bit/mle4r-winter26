import json
import subprocess
import sys
import time
from pathlib import Path

TRAIN_SCRIPT = Path("train_mnist.py")
OUT_ROOT = Path("..") / "artifacts"
DEVICE = "gpu"
TRACK = 0  # 0/1


def main() -> None:
    sweep_dir = OUT_ROOT / "sweeps" / f"sweep__{int(time.time())}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest = sweep_dir / "manifest.jsonl"

    seeds = [1, 2]
    opts = ["adam", "sgd"]
    lrs = [3e-4, 1e-3]
    bs, epochs = 64, 3

    runs = [(seed, opt, lr) for seed in seeds for opt in opts for lr in lrs]
    print(f"sweep_dir: {sweep_dir}")
    print(f"n_runs: {len(runs)}")

    def cmd_for(run_name: str, seed: int, opt: str, lr: float) -> list[str]:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--run-name", run_name,
            "--out-root", str(OUT_ROOT),
            "--seed", str(seed),
            "--optimizer", opt,
            "--learning-rate", str(lr),
            "--batch-size", str(bs),
            "--max-epochs", str(epochs),
            "--device", DEVICE,
            "--track", str(TRACK),
        ]
        if opt == "sgd":
            cmd += ["--momentum", "0.9"]
        return cmd

    with manifest.open("a") as mf:
        for i, (seed, opt, lr) in enumerate(runs, 1):
            run_name = f"mnist__{opt}__seed{seed}__lr{lr}"
            run_dir = OUT_ROOT / "runs" / run_name

            print(f"[{i}/{len(runs)}] {run_name}")

            run_dir.mkdir(parents=True, exist_ok=True)  # for logs
            t0 = time.time()
            p = None

            with (run_dir / "stdout.log").open("w") as out, (run_dir / "stderr.log").open("w") as err:
                cmd = cmd_for(run_name, seed, opt, lr)
                out.write("CMD: " + " ".join(cmd) + "\n")
                out.flush()
                p = subprocess.run(cmd, stdout=out, stderr=err)

            rec = {
                "run_name": run_name,
                "status": "ok" if p.returncode == 0 else "fail",
                "returncode": p.returncode,
                "seconds": time.time() - t0,
                "cfg": {"seed": seed, "opt": opt, "lr": lr},
            }

            mp = run_dir / "metrics.json"
            if p.returncode == 0 and mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    rec["final_test_acc"] = m.get("final_test_acc")
                except Exception:
                    rec["final_test_acc"] = None

            mf.write(json.dumps(rec) + "\n")
            mf.flush()

            if p.returncode != 0:
                print(f"FAIL-FAST. See: {run_dir / 'stderr.log'}")
                sys.exit(p.returncode)

    print(f"Done. Manifest: {manifest}")


if __name__ == "__main__":
    main()
