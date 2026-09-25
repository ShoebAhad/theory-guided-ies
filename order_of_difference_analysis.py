
import os

import math

import matplotlib.pyplot as plt
import numpy as np


def comb(n, k):
    return math.comb(n, k)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DUMP_PATH = os.path.join(RESULTS_DIR, "noise_autocorr_losses.npy")
NEAR_ZERO_THRESHOLD = 0.01
ORDERS = [1, 2, 3, 4]


def main():
    a = np.load(DUMP_PATH)  # (N_inst, T), T=100, may contain NaN for a few instances
    valid_rows = ~np.isnan(a).any(axis=1)
    a = a[valid_rows]
    print(f"Loaded {a.shape[0]} instances x {a.shape[1]} epochs (dropped {(~valid_rows).sum()} with NaN)")

    mean_loss = a.mean(axis=1)
    converged = mean_loss < NEAR_ZERO_THRESHOLD
    print(f"{converged.sum()} instances treated as already-converged (mean loss < {NEAR_ZERO_THRESHOLD})")

    diff2_converged = np.diff(a[converged], n=2, axis=1)
    sigma2_hat = diff2_converged.var() / 6.0
    print(f"sigma^2_hat (from converged-subset Var(Delta^2)/6) = {sigma2_hat:.3e}")

    def report(subset_mask, label):
        sub = a[subset_mask]
        print(f"\n--- {label}: n={sub.shape[0]} ---")
        empirical_mse = []
        noise_floor = []
        for N in ORDERS:
            d = np.diff(sub, n=N, axis=1)
            empirical_mse.append(float(np.mean(d ** 2)))
            noise_floor.append(float(comb(2 * N, N) * sigma2_hat))
        empirical_mse = np.array(empirical_mse)
        noise_floor = np.array(noise_floor)
        signal_component = empirical_mse - noise_floor
        for N, mse, floor, sig in zip(ORDERS, empirical_mse, noise_floor, signal_component):
            print(f"N={N}: empirical MSE={mse:.3e}  predicted noise floor C(2N,N)sigma^2={floor:.3e}  "
                  f"implied signal^2={sig:.3e}")
        best_N = ORDERS[int(np.argmin(empirical_mse))]
        print(f"MSE-minimizing order: N={best_N} "
              f"({'matches' if best_N == 2 else 'DOES NOT match'} prop:mse's N=2 claim)")
        return empirical_mse, noise_floor

    empirical_mse, noise_floor = report(np.ones(a.shape[0], dtype=bool), "ALL instances (population average)")
    borderline = (mean_loss >= NEAR_ZERO_THRESHOLD) & (mean_loss < 0.5)
    report(borderline, f"BORDERLINE instances (still-decaying, {NEAR_ZERO_THRESHOLD} <= mean loss < 0.5)"
                        " -- the SNR band prop:mse actually targets")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ORDERS, empirical_mse, "o-", label="Empirical MSE(N) (real trajectories)")
    ax.plot(ORDERS, noise_floor, "s--", label=r"Predicted noise floor $\binom{2N}{N}\hat\sigma^2$")
    ax.set_xlabel("Finite-difference order $N$")
    ax.set_ylabel("Mean squared $N$-th difference")
    ax.set_yscale("log")
    ax.set_xticks(ORDERS)
    ax.legend()
    ax.set_title("Order-of-difference MSE on real ResNet-18/CIFAR-10 trajectories")
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_order_of_difference.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")


DRIFT_LOSSES_PATH = os.path.join(RESULTS_DIR, "drift_exponential_losses.npy")
CALIBRATED_DELTA_EXPONENTIAL = 0.01597  # paper's own reported value, sec:lrmechanism, epoch-31 recalibration


def refined_near_threshold_test():
    """Item 18b: the exploratory test above used a coarse mean-loss proxy for "still
    decaying". This instead evaluates each order N's finite difference EXACTLY at the
    epoch each instance's own |Delta^2 L_i| first drops below the paper's actual
    calibrated delta (0.01597, exponential schedule) -- the real operating point TG-IES's
    mastery check uses, not a proxy for it. Uses the full 0-200 epoch drift dump (not the
    epoch-101-200-only noise_autocorr dump), so N up to 4 is always computable.
    """
    losses = np.load(DRIFT_LOSSES_PATH)  # (N, 200)
    valid_rows = ~np.isnan(losses).any(axis=1)
    losses = losses[valid_rows]
    delta2 = np.diff(losses, n=2, axis=1)  # (N, 198); delta2[:,k] = L[k+2]-2L[k+1]+L[k]
    below = np.abs(delta2) < CALIBRATED_DELTA_EXPONENTIAL
    has_cross = below.any(axis=1)
    t_cross = np.argmax(below, axis=1)[has_cross] + 2  # index into `losses`' epoch axis
    losses = losses[has_cross]

    print(f"\n=== Refined near-threshold test (item 18b): {has_cross.sum()}/{len(valid_rows)} "
          f"instances cross delta={CALIBRATED_DELTA_EXPONENTIAL} within 200 epochs ===")

    min_history = max(ORDERS) + 1
    usable = t_cross >= min_history
    t_cross, losses_usable = t_cross[usable], losses[usable]
    print(f"{usable.sum()} of those have >= {min_history} epochs of history at their crossing point")

    sq_by_order = {}
    for N in ORDERS:
        vals = np.array([
            np.diff(losses_usable[i, t_cross[i] - N: t_cross[i] + 1], n=N)[-1]
            for i in range(len(t_cross))
        ])
        sq_by_order[N] = float(np.mean(vals ** 2))
        print(f"N={N}: MSE at own crossing epoch = {sq_by_order[N]:.3e}")
    best_N = min(sq_by_order, key=sq_by_order.get)
    print(f"MSE-minimizing order at the ACTUAL decision boundary: N={best_N} "
          f"({'matches' if best_N == 2 else 'DOES NOT match'} prop:mse's N=2 claim)")
    return sq_by_order


def phase_quartile_analysis():
    """Item 4: split the full 0-200 epoch trajectory into quartiles and estimate the
    r_i (local geometric decay rate) distribution per quartile via the first-difference
    ratio r_hat_i(t) = Delta^1 L_i(t+1) / Delta^1 L_i(t) (ass:geometric's own r_i, thm:ratio's
    ratio statistic, empirically estimated rather than assumed).
    """
    losses = np.load(DRIFT_LOSSES_PATH)
    valid_rows = ~np.isnan(losses).any(axis=1)
    losses = losses[valid_rows]
    d1 = np.diff(losses, n=1, axis=1)  # (N, 199)
    epochs = losses.shape[1]

    print("\n=== Training-phase quartile analysis of r_i (item 4) ===")
    quartiles = [(0, 50), (50, 100), (100, 150), (150, epochs - 1)]
    results = []
    for (lo, hi) in quartiles:
        num = d1[:, lo + 1:hi]
        den = d1[:, lo:hi - 1]
        both_decreasing = (num < 0) & (den < 0)
        valid = both_decreasing & (np.abs(den) > 1e-5)
        r_hat = np.where(valid, num / den, np.nan)
        r_hat = np.clip(r_hat, 0, 1.5)
        flat = r_hat[valid]
        mean, std = np.nanmean(flat), np.nanstd(flat)
        cv = std / mean if mean != 0 else np.nan
        p5, p50, p95 = np.percentile(flat, [5, 50, 95])
        print(f"epochs [{lo+1},{hi}]: n={len(flat)}  mean(r)={mean:.3f}  std={std:.3f}  "
              f"CV={cv:.3f}  p5={p5:.3f}  p50={p50:.3f}  p95={p95:.3f}")
        results.append(dict(label=f"{lo+1}-{hi}", flat=flat, mean=mean, p5=p5, p50=p50, p95=p95))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot([r["flat"] for r in results], tick_labels=[r["label"] for r in results], showfliers=False)
    ax.set_xlabel("Training-epoch quartile")
    ax.set_ylabel(r"Empirical $\hat r_i$ (first-difference ratio)")
    ax.set_title("Local decay-rate distribution across training phases")
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_phase_quartile_r.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    main()
    refined_near_threshold_test()
    phase_quartile_analysis()
