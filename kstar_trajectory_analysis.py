
import csv
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
KSTARS = [1, 2, 4, 8, 15, 30]
SEEDS = [0, 1, 2]
SETTING = "cifar10_resnet18_SGD_exponential_learning_rate"

COLORS = {1: "#CC2936", 2: "#E69F00", 4: "#F0C808", 8: "#009E73", 15: "#56B4E9", 30: "#0A2C55"}


def load_run_summary(kstar, seed):
    path = os.path.join(RESULTS_DIR, f"tgies_{SETTING}_seed{seed}_kstar{kstar}.csv")
    rows = list(csv.DictReader(open(path)))
    N = int(rows[0]["active_set_size"])
    epochs = len(rows)
    best_acc = max(float(r["test_acc_pct"]) for r in rows)
    cum_bp = int(rows[-1]["cum_backprop_instances"])
    saved = 100.0 * (1 - cum_bp / (N * epochs))
    return best_acc, saved


def load_traj(kstar, seed):
    path = os.path.join(RESULTS_DIR, f"traj_dist_tgies_{SETTING}_seed{seed}_kstar{kstar}.csv")
    return list(csv.DictReader(open(path)))


def load_baseline_acc():
    accs = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"baseline_{SETTING}_seed{seed}_traj.csv")
        rows = list(csv.DictReader(open(path)))
        accs.append(max(float(r["test_acc_pct"]) for r in rows))
    return statistics.mean(accs), statistics.stdev(accs)


def main():
    baseline_acc_mean, baseline_acc_sd = load_baseline_acc()
    print(f"Matched-batch baseline (3-seed, this instrumentation batch): "
          f"{baseline_acc_mean:.2f}+-{baseline_acc_sd:.2f}")

    print(f"\n{'K*':>4} {'n':>2} {'acc mean':>9} {'acc sd':>7} {'saved mean':>11} {'saved sd':>9} "
          f"{'acc gap vs baseline':>20}")
    sweep_rows = []
    for k in KSTARS:
        accs, saveds = [], []
        for s in SEEDS:
            acc, saved = load_run_summary(k, s)
            accs.append(acc)
            saveds.append(saved)
        acc_m, acc_s = statistics.mean(accs), statistics.stdev(accs)
        sav_m, sav_s = statistics.mean(saveds), statistics.stdev(saveds)
        gap = acc_m - baseline_acc_mean
        print(f"{k:>4} {3:>2} {acc_m:>9.2f} {acc_s:>7.2f} {sav_m:>11.2f} {sav_s:>9.2f} {gap:>20.2f}")
        sweep_rows.append((k, acc_m, acc_s, sav_m, sav_s))

    print(f"\n{'K*':>4}  epoch-by-epoch mean absolute param L2 distance from shared baseline reference")
    traj_abs = {}
    traj_rel = {}
    ref_norm = {}
    for k in KSTARS:
        by_epoch_abs = defaultdict(list)
        by_epoch_rel = defaultdict(list)
        by_epoch_ref = defaultdict(list)
        for s in SEEDS:
            for r in load_traj(k, s):
                e = int(r["epoch"])
                by_epoch_abs[e].append(float(r["param_l2_dist"]))
                by_epoch_rel[e].append(float(r["rel_dist"]))
                by_epoch_ref[e].append(float(r["ref_param_l2_norm"]))
        epochs_sorted = sorted(by_epoch_abs.keys())
        traj_abs[k] = (epochs_sorted, [statistics.mean(by_epoch_abs[e]) for e in epochs_sorted])
        traj_rel[k] = (epochs_sorted, [statistics.mean(by_epoch_rel[e]) for e in epochs_sorted])
        ref_norm[k] = (epochs_sorted, [statistics.mean(by_epoch_ref[e]) for e in epochs_sorted])
        print(f"{k:>4}  abs={['%.1f' % v for v in traj_abs[k][1]]}")
        print(f"      rel={['%.3f' % v for v in traj_rel[k][1]]}")

    # sanity check: reference (baseline) norm should be ~K-independent (same 3 baseline
    # runs referenced by every K*), report the spread across K to confirm
    ref_final = [ref_norm[k][1][-1] for k in KSTARS]
    print(f"\nSanity check: epoch-200 reference norm across K* values (should be near-identical): "
          f"{['%.2f' % v for v in ref_final]}")

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for k in KSTARS:
        epochs, vals = traj_abs[k]
        ax1.plot(epochs, vals, color=COLORS[k], marker="o", markersize=5, linewidth=2.2,
                  label=f"$K^*={k}$")
        epochs_r, vals_r = traj_rel[k]
        ax2.plot(epochs_r, vals_r, color=COLORS[k], marker="o", markersize=5, linewidth=2.2,
                  label=f"$K^*={k}$")
    ax1.set_xlabel("Epoch", fontsize=15)
    ax1.set_ylabel(r"$\|\theta^{(t)}_{\mathrm{TG\text{-}IES}} - \theta^{(t)}_{\mathrm{baseline}}\|$", fontsize=15)
    ax1.set_title("Absolute parameter distance", fontsize=16)
    ax1.tick_params(labelsize=13)
    ax1.legend(fontsize=11, ncol=2)
    ax2.set_xlabel("Epoch", fontsize=15)
    ax2.set_ylabel("Relative distance (normalized by reference norm)", fontsize=15)
    ax2.set_title("Relative parameter distance", fontsize=16)
    ax2.tick_params(labelsize=13)
    ax2.legend(fontsize=11, ncol=2)
    fig.suptitle(r"Trajectory divergence from the shared full-data baseline reference, "
                 r"primary cell, 3-seed mean", fontsize=17, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(FIG_DIR, "fig_kstar_trajectory.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved {out_path}")
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})


if __name__ == "__main__":
    main()
