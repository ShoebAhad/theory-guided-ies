
import csv
import glob
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO, "results")
SCRATCH = "/tmp/claude-1000/-home-milgpt/24e68007-a066-4ca6-8810-67a765ea599a/scratchpad/memtest"
METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies"]
PRIMARY = dict(dataset="cifar10", model="resnet18",
               optimizer="SGD_exponential_learning_rate", seed=0)
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


def run_and_poll(method):
    os.makedirs(SCRATCH, exist_ok=True)
    tag = "efftest"
    cmd = [
        sys.executable, os.path.join(REPO, "train.py"),
        "--method", method, "--dataset", PRIMARY["dataset"], "--model", PRIMARY["model"],
        "--optimizer", PRIMARY["optimizer"], "--seed", str(PRIMARY["seed"]),
        "--epochs", str(SHORT_EPOCHS), "--recalib_every", str(RECALIB_EVERY),
        "--num_workers", "2", "--out_dir", SCRATCH, "--tag", tag,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []  # (mem_mib, util_pct, power_w)
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
        raise RuntimeError(f"{method} run failed (see output above)")
    return samples, elapsed, out


def aggregate_wallclock():
    """Real wall-clock time from the existing full 200-epoch runs (epoch_time_s
    column), primary CIFAR-10/ResNet-18/SGD-exponential setting, all seeds present."""
    rows = []
    for method in METHODS:
        for seed in (0, 1, 2):
            path = os.path.join(
                RESULTS_DIR,
                f"{method}_cifar10_resnet18_SGD_exponential_learning_rate_seed{seed}.csv")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = list(csv.DictReader(f))
            if not r:
                continue
            total_s = sum(float(row["epoch_time_s"]) for row in r)
            per_epoch = total_s / len(r)
            best_acc = max(float(row["test_acc_pct"]) for row in r)  # matches paper convention (sec:setup)
            saved_pct = 100.0 * (1 - int(r[-1]["cum_backprop_instances"]) /
                                  (int(r[0]["active_set_size"]) * len(r)))
            rows.append(dict(method=method, seed=seed, epochs=len(r), total_wallclock_s=total_s,
                              per_epoch_s=per_epoch, best_test_acc_pct=best_acc,
                              backprop_saved_pct=saved_pct))
    return rows


def main():
    print("=" * 70)
    print("Part 1: real wall-clock time from existing full 200-epoch runs")
    print("=" * 70)
    wc_rows = aggregate_wallclock()
    wc_path = os.path.join(RESULTS_DIR, "wallclock_summary.csv")
    with open(wc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wc_rows[0].keys()))
        w.writeheader()
        w.writerows(wc_rows)
    by_method = {}
    for row in wc_rows:
        by_method.setdefault(row["method"], []).append(row)
    for method in METHODS:
        rs = by_method.get(method, [])
        if not rs:
            print(f"  {method}: no full-run data found"); continue
        mean_total = statistics.mean(r["total_wallclock_s"] for r in rs)
        mean_acc = statistics.mean(r["best_test_acc_pct"] for r in rs)
        mean_saved = statistics.mean(r["backprop_saved_pct"] for r in rs)
        print(f"  {method:12s} n_seeds={len(rs)}  mean_wallclock={mean_total/60:.2f} min "
              f"acc={mean_acc:.2f}%  backprop_saved={mean_saved:.1f}%")
    print()

    print("=" * 70)
    print(f"Part 2: peak memory / utilization / power, short {SHORT_EPOCHS}-epoch "
          f"instrumented runs (recalib_every={RECALIB_EVERY})")
    print("=" * 70)
    idle_mem, idle_util, idle_pow = sample_idle_baseline()
    print(f"  idle GPU baseline (other process): mem={idle_mem:.0f} MiB "
          f"util={idle_util:.0f}% power={idle_pow:.1f} W")

    eff_rows = []
    for method in METHODS:
        print(f"  running {method} ({SHORT_EPOCHS} epochs)...")
        samples, elapsed, out = run_and_poll(method)
        mems = [s[0] for s in samples]
        utils = [s[1] for s in samples]
        pows = [s[2] for s in samples]
        peak_mem_delta = max(mems) - idle_mem if mems else float("nan")
        mean_util = statistics.mean(utils) if utils else float("nan")
        mean_pow = statistics.mean(pows) if pows else float("nan")
        # energy for this short instrumented run itself (informational)
        short_run_energy_wh = mean_pow * elapsed / 3600.0
        eff_rows.append(dict(
            method=method, n_samples=len(samples), short_run_wallclock_s=elapsed,
            peak_mem_used_delta_mib=peak_mem_delta, mean_gpu_util_pct=mean_util,
            mean_power_w=mean_pow, short_run_energy_wh=short_run_energy_wh,
        ))
        print(f"    peak_mem_delta={peak_mem_delta:.0f} MiB  mean_util={mean_util:.1f}%  "
              f"mean_power={mean_pow:.1f} W  elapsed={elapsed:.1f}s")

    eff_path = os.path.join(RESULTS_DIR, "efficiency_summary.csv")
    with open(eff_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
        w.writeheader()
        w.writerows(eff_rows)

    print()
    print("=" * 70)
    print("Part 3: full-run energy estimate (short-run mean power x real full-run wallclock)")
    print("=" * 70)
    combined_rows = []
    eff_by_method = {r["method"]: r for r in eff_rows}
    for method in METHODS:
        rs = by_method.get(method, [])
        if not rs or method not in eff_by_method:
            continue
        mean_total_s = statistics.mean(r["total_wallclock_s"] for r in rs)
        mean_pow = eff_by_method[method]["mean_power_w"]
        est_energy_wh = mean_pow * mean_total_s / 3600.0
        combined_rows.append(dict(
            method=method, mean_full_run_wallclock_min=mean_total_s / 60.0,
            mean_power_w=mean_pow, peak_mem_used_delta_mib=eff_by_method[method]["peak_mem_used_delta_mib"],
            mean_gpu_util_pct=eff_by_method[method]["mean_gpu_util_pct"],
            est_full_run_energy_wh=est_energy_wh,
        ))
        print(f"  {method:12s} wallclock={mean_total_s/60:.2f} min  power~{mean_pow:.1f} W  "
              f"peak_mem_delta~{eff_by_method[method]['peak_mem_used_delta_mib']:.0f} MiB  "
              f"est_energy~{est_energy_wh:.1f} Wh")

    combined_path = os.path.join(RESULTS_DIR, "efficiency_combined_summary.csv")
    with open(combined_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        w.writeheader()
        w.writerows(combined_rows)

    print()
    print(f"Wrote {wc_path}\n      {eff_path}\n      {combined_path}")


if __name__ == "__main__":
    main()
