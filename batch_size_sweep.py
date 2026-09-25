
import csv
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO, "results")
SCRATCH = os.path.join(REPO, "_batch_size_sweep_scratch")
METHODS = ["baseline", "ies_shipped", "tgies"]
BATCH_SIZES = [32, 64, 128, 256]
PRIMARY = dict(dataset="cifar10", model="resnet18", optimizer="SGD_exponential_learning_rate", seed=0)
SHORT_EPOCHS = 8
RECALIB_EVERY = 5
POLL_INTERVAL_S = 0.5


def query_gpu():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    mem_mib, util_pct, power_w = out.split(",")
    return float(mem_mib), float(util_pct), float(power_w)


def sample_idle_baseline(n=6, interval=0.5):
    mems, utils, pows = [], [], []
    for _ in range(n):
        m, u, p = query_gpu()
        mems.append(m); utils.append(u); pows.append(p)
        time.sleep(interval)
    return statistics.median(mems), statistics.median(utils), statistics.median(pows)


def run_and_poll(method, batch_size):
    os.makedirs(SCRATCH, exist_ok=True)
    tag = f"bssweep_bs{batch_size}"
    cmd = [
        sys.executable, os.path.join(REPO, "train.py"),
        "--method", method, "--dataset", PRIMARY["dataset"], "--model", PRIMARY["model"],
        "--optimizer", PRIMARY["optimizer"], "--seed", str(PRIMARY["seed"]),
        "--epochs", str(SHORT_EPOCHS), "--recalib_every", str(RECALIB_EVERY),
        "--batch_size", str(batch_size),
        "--num_workers", "4", "--out_dir", SCRATCH, "--tag", tag, "--ckpt_every", "0",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []
    t0 = time.time()
    while proc.poll() is None:
        try:
            samples.append(query_gpu())
        except subprocess.CalledProcessError:
            pass
        time.sleep(POLL_INTERVAL_S)
    out, _ = proc.communicate()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(out)
        raise RuntimeError(f"{method}/bs={batch_size} run failed (see output above)")

    csv_path = os.path.join(SCRATCH, f"{method}_{PRIMARY['dataset']}_{PRIMARY['model']}_"
                                      f"{PRIMARY['optimizer']}_seed{PRIMARY['seed']}_{tag}.csv")
    rows = list(csv.DictReader(open(csv_path)))
    N = int(rows[0]["active_set_size"])
    epochs = len(rows)
    cum_bp = int(rows[-1]["cum_backprop_instances"])
    backprop_saved_pct = 100.0 * (1 - cum_bp / (N * epochs))
    mean_epoch_time = statistics.mean(float(r["epoch_time_s"]) for r in rows)
    return samples, elapsed, backprop_saved_pct, mean_epoch_time


def main():
    idle_mem, idle_util, idle_pow = sample_idle_baseline()
    print(f"idle GPU baseline: mem={idle_mem:.0f} MiB util={idle_util:.0f}% power={idle_pow:.1f} W\n")

    rows_out = []
    for bs in BATCH_SIZES:
        for method in METHODS:
            print(f"running {method} bs={bs} ({SHORT_EPOCHS} epochs, recalib_every={RECALIB_EVERY})...")
            samples, elapsed, backprop_saved_pct, mean_epoch_time = run_and_poll(method, bs)
            mems = [s[0] for s in samples]
            utils = [s[1] for s in samples]
            pows = [s[2] for s in samples]
            peak_mem_delta = max(mems) - idle_mem if mems else float("nan")
            mean_util = statistics.mean(utils) if utils else float("nan")
            mean_pow = statistics.mean(pows) if pows else float("nan")
            rows_out.append(dict(
                method=method, batch_size=bs, n_samples=len(samples),
                peak_mem_used_delta_mib=peak_mem_delta, mean_gpu_util_pct=mean_util,
                mean_power_w=mean_pow, mean_epoch_time_s=mean_epoch_time,
                backprop_saved_pct_short_run=backprop_saved_pct,
            ))
            print(f"  peak_mem_delta={peak_mem_delta:.0f} MiB  mean_util={mean_util:.1f}%  "
                  f"mean_epoch_time={mean_epoch_time:.2f}s  "
                  f"backprop_saved(8ep)={backprop_saved_pct:.2f}%")

    out_path = os.path.join(RESULTS_DIR, "batch_size_sweep_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
