import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
SETTINGS = [
    ("cifar10_resnet18", "cifar10_resnet18_SGD_exponential_learning_rate_seed0"),
    ("cifar100_resnet18", "cifar100_resnet18_SGD_exponential_learning_rate_seed0"),
    ("cifar10_vgg16", "cifar10_vgg16_SGD_exponential_learning_rate_seed0"),
]
METHODS = ["baseline", "ies_shipped", "ies_alg1", "tgies", "infobatch", "bls_infobatch", "alignprune"]
N_TRAIN = 50000
BUDGETS_PCT = [20, 30, 40, 50]

LABELS = {
    "baseline": "Baseline", "ies_shipped": "IES (shipped)", "ies_alg1": "IES (literal)",
    "tgies": "TG-IES (ours)", "infobatch": "InfoBatch", "bls_infobatch": "BLS-InfoBatch",
    "alignprune": "AlignPrune",
}
# Kept in sync with figures/make_figures.py and figures/make_newbaselines_figure.py's
# COLORS for these same seven methods -- distinct hues, not a matplotlib default
# tab10 cycle, and each paired with its own linestyle/width so overlapping curves in
# the busy low-budget region stay separable without relying on color alone.
COLORS = {
    "baseline": "#8C8C8C", "ies_shipped": "#E69F00", "ies_alg1": "#009E73", "tgies": "#0A2C55",
    "infobatch": "#56B4E9", "bls_infobatch": "#CC2936", "alignprune": "#7A3E9D",
}
STYLES = {
    "baseline": ":", "ies_shipped": "-.", "ies_alg1": "--", "tgies": "-",
    "infobatch": "--", "bls_infobatch": "-.", "alignprune": ":",
}
WIDTHS = {
    "baseline": 1.3, "ies_shipped": 1.5, "ies_alg1": 1.5, "tgies": 2.6,
    "infobatch": 1.5, "bls_infobatch": 1.5, "alignprune": 1.5,
}
ZORDER = {"tgies": 10}


def load(method, setting):
    path = os.path.join(RESULTS_DIR, f"{method}_{setting}.csv")
    if not os.path.exists(path):
        return None
    epochs, cum_bp, acc = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            cum_bp.append(int(row["cum_backprop_instances"]))
            acc.append(float(row["test_acc_pct"]))
    return np.array(epochs), np.array(cum_bp), np.array(acc)


def analyze_setting(setting):
    curves = {}
    for m in METHODS:
        d = load(m, setting)
        if d is None:
            continue
        epochs, cum_bp, acc = d
        total_epochs = epochs.max()
        compute_pct = 100.0 * cum_bp / (N_TRAIN * total_epochs)  # % of the FULL 200-epoch budget spent so far
        best_acc_so_far = np.maximum.accumulate(acc)
        curves[m] = (compute_pct, best_acc_so_far, acc)
    return curves


def main():
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(19.5, 5.6), sharey=False)
    if len(SETTINGS) == 1:
        axes = [axes]

    for ax, (label, setting) in zip(axes, SETTINGS):
        curves = analyze_setting(setting)
        print(f"\n=== {label} ===")
        print(f"{'method':16s} " + " ".join(f"acc@{b}%" for b in BUDGETS_PCT) + "   final_acc  best_acc  gap")
        # draw tgies last (on top) so it isn't buried under the other six curves
        # in the busy, tightly-overlapping low-budget region
        for m, (compute_pct, best_so_far, acc) in sorted(curves.items(), key=lambda kv: kv[0] == "tgies"):
            row = []
            for b in BUDGETS_PCT:
                if compute_pct[-1] < b:
                    row.append(np.nan)
                else:
                    row.append(float(np.interp(b, compute_pct, best_so_far)))
            final_acc = acc[-1]
            best_acc = acc.max()
            print(f"{m:16s} " + " ".join(f"{v:6.2f}" if not np.isnan(v) else "   n/a" for v in row) +
                  f"   {final_acc:7.2f}  {best_acc:7.2f}  {best_acc - final_acc:5.2f}")
            ax.plot(compute_pct, best_so_far, label=LABELS[m], color=COLORS[m], linestyle=STYLES[m],
                     linewidth=WIDTHS[m], alpha=1.0 if m == "tgies" else 0.85, zorder=ZORDER.get(m, 5))

        ax.set_xlabel("Cumulative backprop instances used (% of N x epochs)", fontsize=15)
        ax.set_ylabel("Best test accuracy so far (%)", fontsize=15)
        # the informative region (where methods separate) is the first ~40% of
        # budget; every curve is already flat/converged well before 100%, so the
        # full 0-100 range mostly wastes width on an indistinguishable tail
        ax.set_xlim(0, 45)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=11, ncol=2, loc="lower right")
        ax.set_title(label, fontsize=16)

    fig.suptitle("Accuracy vs. compute budget (0-45% shown; all curves are flat beyond this), "
                  "all cells with new-baseline data", fontsize=17)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_equal_compute.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved {out_path}")
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})


if __name__ == "__main__":
    main()
