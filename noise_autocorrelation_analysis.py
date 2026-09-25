
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DUMP_PATH = os.path.join(RESULTS_DIR, "noise_autocorr_losses.npy")
MAX_LAG = 5


def detrend(traj):
    """traj: (T,) with no NaNs. OLS linear detrend (see module docstring for why
    not a moving average): residual = traj - (a + b*t), fit by least squares."""
    T = len(traj)
    if T < 10:
        return np.array([])
    t = np.arange(T, dtype=np.float64)
    b, a = np.polyfit(t, traj, deg=1)
    resid = traj - (a + b * t)
    return resid


def pooled_lag_corr(residuals_list, lag):
    xs, ys = [], []
    for r in residuals_list:
        if len(r) <= lag:
            continue
        xs.append(r[:-lag] if lag > 0 else r)
        ys.append(r[lag:] if lag > 0 else r)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 0.0, len(x)


def main():
    if not os.path.exists(DUMP_PATH):
        raise SystemExit(f"{DUMP_PATH} not found -- run noise_autocorrelation_run.py first")
    losses = np.load(DUMP_PATH, mmap_mode="r")  # (N, T)
    N, T = losses.shape
    print(f"Loaded {DUMP_PATH}: N={N} instances, T={T} epochs")

    residuals_list = []
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(N, size=min(N, 20000), replace=False)  # subsample for speed
    for i in sample_idx:
        traj = np.asarray(losses[i], dtype=np.float64)
        if np.isnan(traj).any():
            continue
        residuals_list.append(detrend(traj))

    print(f"Detrended {len(residuals_list)} instance trajectories (OLS linear)")

    all_resid = np.concatenate(residuals_list)
    sigma_hat = float(all_resid.std())
    print(f"Pooled residual std sigma_hat = {sigma_hat:.6f}  (n={all_resid.size})")

    lags = list(range(0, MAX_LAG + 1))
    rhos, npairs = [], []
    for lag in lags:
        rho, npair = pooled_lag_corr(residuals_list, lag)
        rhos.append(rho)
        npairs.append(npair)
        print(f"  lag {lag}: rho={rho:+.4f}  (n_pairs={npair})")

    rho1, rho2 = rhos[1], rhos[2]
    naive_var = 6.0 * sigma_hat ** 2
    corrected_var = (6.0 - 8.0 * rho1 + 2.0 * rho2) * sigma_hat ** 2
    correction_factor = 6.0 - 8.0 * rho1 + 2.0 * rho2
    print(f"\nNaive Var(Delta^2 xi) assuming independence = 6*sigma^2 = {naive_var:.6e}")
    print(f"Corrected Var(Delta^2 xi) with measured rho1,rho2 = "
          f"(6 - 8*rho1 + 2*rho2)*sigma^2 = {correction_factor:.4f}*sigma^2 = {corrected_var:.6e}")
    print(f"Ratio corrected/naive = {corrected_var/naive_var:.4f}")

    # 95% null band for zero autocorrelation, using the smallest pair count (most conservative)
    ci = 1.96 / np.sqrt(min(npairs[1:]))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    ax.bar(lags, rhos, color="#1F5C99", width=0.5, zorder=3)
    ax.axhspan(-ci, ci, color="0.85", alpha=0.6, zorder=1, label=f"95% null band (±{ci:.3f})")
    ax.axhline(0, color="0.3", linewidth=0.8, zorder=2)
    ax.set_xlabel("Lag (epochs)", fontweight="bold")
    ax.set_ylabel("Pooled autocorrelation " + r"$\hat\rho_k$", fontweight="bold")
    ax.set_xticks(lags)
    ax.legend(fontsize=6, frameon=True, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG_DIR, "fig4_noise_autocorr.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_noise_autocorr.png"), dpi=300)
    print(f"\nSaved {FIG_DIR}/fig4_noise_autocorr.{{pdf,png}}")

    with open(os.path.join(RESULTS_DIR, "noise_autocorr_summary.txt"), "w") as f:
        f.write(f"N_instances_used={len(residuals_list)}\nT_epochs={T}\ndetrend=OLS_linear\n")
        f.write(f"sigma_hat={sigma_hat:.6f}\n")
        for lag, rho, npair in zip(lags, rhos, npairs):
            f.write(f"rho_lag{lag}={rho:.6f} (n_pairs={npair})\n")
        f.write(f"95%_null_band=+-{ci:.6f}\n")
        f.write(f"naive_var_6sigma2={naive_var:.6e}\n")
        f.write(f"corrected_var_factor={correction_factor:.4f}\n")
        f.write(f"corrected_var={corrected_var:.6e}\n")
        f.write(f"ratio_corrected_over_naive={corrected_var/naive_var:.4f}\n")
    print(f"Saved {RESULTS_DIR}/noise_autocorr_summary.txt")


if __name__ == "__main__":
    main()
