
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
FIG_DIR = "figures"
N = 50000
TAU = 0.02
BETA_TEST = 1.0
ETA0, GAMMA = 0.1, 0.96
ETA_31 = ETA0 * GAMMA ** 31  # matches tab:ksafe's convention exactly

SEEDS = [0, 1, 2]
TAG = "reactprobe2"


def main():
    react_frames, main_frames = [], []
    for s in SEEDS:
        base = f"tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed{s}_{TAG}"
        r = pd.read_csv(os.path.join(RESULTS_DIR, f"reactivation_{base}.csv"))
        r["seed"] = s
        react_frames.append(r)
        m = pd.read_csv(os.path.join(RESULTS_DIR, f"{base}.csv"))
        m["seed"] = s
        m["pi_t"] = m["total_excluded_now"] / N
        main_frames.append(m)
    react = pd.concat(react_frames, ignore_index=True)
    main_df = pd.concat(main_frames, ignore_index=True)

    react["cancel_ratio"] = react["agg_grad_norm"] / react["mean_gnorm"]

    # attach pi_t (exclusion fraction) at each probed epoch, per seed
    react = react.merge(main_df[["seed", "epoch", "pi_t"]], on=["seed", "epoch"], how="left")

    print("=== Per-epoch (pooled across 3 seeds): mean_gnorm, agg_grad_norm, cancel_ratio, eps0, pi_t ===")
    g = react.groupby("epoch").agg(
        mean_gnorm=("mean_gnorm", "mean"),
        agg_grad_norm=("agg_grad_norm", "mean"),
        cancel_ratio=("cancel_ratio", "mean"),
        eps0=("eps0", "mean"),
        pi_t=("pi_t", "mean"),
        frac_exceed=("frac_exceed_eps0", "mean"),
    )
    print(g.round(5).to_string())

    overall_cancel = react["cancel_ratio"].mean()
    print(f"\nOverall mean cancellation ratio ||mu_t|| / mean_i||g_i|| = {overall_cancel:.3f} "
          f"(pooled over {len(react)} (epoch,seed) points)")

    # Aggregate floor bar-epsilon: use the max observed ||mu_t|| across all probed
    # epochs/seeds (the honest sup over the horizon this assumption is meant to hold on,
    # not a favorable mean) -- matches how thm:safe/cor:kstar's own eps0 is a floor, not
    # a typical value.
    eps_bar = react["agg_grad_norm"].max()
    pi_bar = react["pi_t"].max()
    eps0_mean = react["eps0"].mean()
    print(f"\nbar_eps (max agg_grad_norm over horizon) = {eps_bar:.5f}")
    print(f"bar_pi  (max exclusion fraction pi_t)     = {pi_bar:.5f}")
    print(f"mean eps0 (per-instance floor, for comparison) = {eps0_mean:.5f}")

    k_safe_agg = TAU / (BETA_TEST * ETA_31 * pi_bar * eps_bar)
    k_safe_per_instance_ref = N * TAU / (BETA_TEST * ETA_31 * eps0_mean)  # tab:ksafe-style, for comparison
    print(f"\nK_safe^agg (cor:kstaragg, floor(.) omitted for readability) = {k_safe_agg:,.1f}")
    print(f"K_safe (per-instance, tab:ksafe-style, using same eta_31)    = {k_safe_per_instance_ref:,.0f}")
    print(f"Actual K* used everywhere in this paper                     = 15")
    print(f"K_safe^agg / K*_used ratio                                  = {k_safe_agg/15:.2f}x")

    # ---- figure ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))

    ax = axes[0]
    epochs = g.index.values
    for s in SEEDS:
        sub = react[react["seed"] == s].sort_values("epoch")
        ax.plot(sub["epoch"], sub["mean_gnorm"], color="#C2571B", alpha=0.35, linewidth=1.0,
                marker="o", markersize=2.5)
        ax.plot(sub["epoch"], sub["agg_grad_norm"], color="#0A2C55", alpha=0.35, linewidth=1.0,
                marker="^", markersize=2.5)
    ax.plot(epochs, g["mean_gnorm"], color="#C2571B", linewidth=2.0, marker="o", markersize=4,
            label=r"mean$_i\|g_i^{(t)}\|$ (per-instance)")
    ax.plot(epochs, g["agg_grad_norm"], color="#0A2C55", linewidth=2.0, marker="^", markersize=4,
            label=r"$\|\mu_t\|$ (aggregate)")
    ax.plot(epochs, g["eps0"], color="0.4", linestyle="--", linewidth=1.2, label=r"$\varepsilon_0$")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Gradient norm", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    ax = axes[1]
    for s in SEEDS:
        sub = react[react["seed"] == s].sort_values("epoch")
        ax.plot(sub["epoch"], sub["cancel_ratio"], alpha=0.4, linewidth=1.0, marker="o", markersize=3,
                color="#009E73")
    ax.plot(epochs, g["cancel_ratio"], color="#009E73", linewidth=2.2, marker="o", markersize=5,
            label="3-seed mean")
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(r"Cancellation ratio $\|\mu_t\|\,/\,$mean$_i\|g_i\|$", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, "fig_aggregate_bound.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_aggregate_bound.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved {FIG_DIR}/fig_aggregate_bound.{{pdf,png}}")


if __name__ == "__main__":
    main()
