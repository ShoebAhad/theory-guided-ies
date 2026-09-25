import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
SCHEDULES = ["exponential", "fixed", "Adam"]
# Each schedule's own final calibrated delta, read from that schedule's primary-cell
# tgies_..._seed0.csv (last row's "delta" column) -- NOT computed in drift_term_run.py
# itself, which trains method=baseline (no calibration) so the comparison uses the real
# calibration a TG-IES run on the same schedule/seed actually reaches.
CALIBRATED_DELTA = {
    "exponential": 0.01596660159042279,   # tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed0.csv
    "fixed": 0.002468162927095511,        # tgies_cifar10_resnet18_SGD_fixed_learning_rate_seed0.csv
    "Adam": 0.0002569255708246006,        # tgies_cifar10_resnet18_Adam_seed0.csv
}


def load(tag):
    losses = np.load(os.path.join(RESULTS_DIR, f"drift_{tag}_losses.npy"))
    inner = np.load(os.path.join(RESULTS_DIR, f"drift_{tag}_inner.npy"))
    subset_idx = np.load(os.path.join(RESULTS_DIR, f"drift_{tag}_subset_idx.npy"))
    eta = np.load(os.path.join(RESULTS_DIR, f"drift_{tag}_eta.npy"))
    norms_path = os.path.join(RESULTS_DIR, f"drift_{tag}_norms.npy")
    norms = np.load(norms_path) if os.path.exists(norms_path) else None
    return losses, inner, subset_idx, eta, norms


def analyze(tag, epoch_min=2, label="all epochs"):
    losses, inner, subset_idx, eta, _norms = load(tag)
    epochs = losses.shape[1]
    sub_losses = losses[subset_idx]  # (subset_size, epochs)

    # Index alignment (verified against drift_term_run.py's semantics + thm:curvature's
    # derivation, not assumed): losses[:,k] and inner[:,k] are both evaluated at the SAME
    # frozen theta_k = "parameters after training epoch k" = paper's theta^(k+1). So
    # L_i^(m)[paper] = losses[:,m-1] and <g_i,g>^(m)[paper] = inner[:,m-1]. Writing the
    # paper's Delta^2 L_i^(t) = L_i^(t)-2L_i^(t-1)+L_i^(t-2) and D_i^(t) =
    # <g_i,g>^(t-2)-<g_i,g>^(t-1) in terms of array index k=t-1 gives:
    #   Delta^2 L_i^(t) = losses[:,k]   - 2*losses[:,k-1] + losses[:,k-2]
    #   D_i^(t)         = inner[:,k-2]  - inner[:,k-1]
    #   eta_t[paper]    = eta[k+1]   (eta_s[paper] = the LR recorded after epoch s, which
    #                                  is the LR used for the step theta^(s)->theta^(s+1);
    #                                  eta_t multiplies D_i^(t), so we need eta at s=t,
    #                                  i.e. eta[t] = eta[k+1] since t=k+1)
    valid_k = [k for k in range(max(2, epoch_min), epochs - 1) if not np.isnan(eta[k + 1])]
    D = inner[:, [k - 2 for k in valid_k]] - inner[:, [k - 1 for k in valid_k]]  # (subset, T)
    delta2L = (sub_losses[:, valid_k] - 2 * sub_losses[:, [k - 1 for k in valid_k]]
               + sub_losses[:, [k - 2 for k in valid_k]])
    eta_t = eta[[k + 1 for k in valid_k]]  # (T,)
    predicted = eta_t[None, :] * D  # eta_t * D_i^(t), broadcast over subset

    actual_flat = delta2L.flatten()
    pred_flat = predicted.flatten()
    mask = np.isfinite(actual_flat) & np.isfinite(pred_flat)
    actual_flat, pred_flat = actual_flat[mask], pred_flat[mask]

    corr = np.corrcoef(actual_flat, pred_flat)[0, 1]
    residual = actual_flat - pred_flat
    mean_abs_drift = np.mean(np.abs(pred_flat))
    mean_abs_residual = np.mean(np.abs(residual))
    slope = np.polyfit(pred_flat, actual_flat, 1)[0]

    print(f"\n=== {tag} ({label}) ===")
    print(f"n={len(actual_flat)} (subset x valid-epoch pairs)")
    print(f"corr(Delta^2 L_i, eta_t*D_i) = {corr:.4f}")
    print(f"slope (actual ~ predicted)  = {slope:.4f}  (1.0 = drift term exactly predicts signal)")
    print(f"mean|eta_t*D_i| (drift term)     = {mean_abs_drift:.3e}")
    print(f"mean|residual| (curvature+rem.)  = {mean_abs_residual:.3e}")
    print(f"drift/residual ratio             = {mean_abs_drift / mean_abs_residual:.2f}x "
          f"({'drift dominates' if mean_abs_drift > mean_abs_residual else 'residual dominates'})")

    return dict(tag=tag, actual=actual_flat, predicted=pred_flat, corr=corr, slope=slope,
                mean_abs_drift=mean_abs_drift, mean_abs_residual=mean_abs_residual)


def analyze_norms(tag, epoch_min=2):
    """Item #17: ||g_i^(t)|| vs |Delta^2 L_i^(t)|, using the SAME index alignment as
    analyze() (norms[:,k] is evaluated at the same frozen theta_k as losses[:,k])."""
    losses, _inner, subset_idx, eta, norms = load(tag)
    if norms is None:
        print(f"skip {tag}: no drift_{tag}_norms.npy")
        return None
    epochs = losses.shape[1]
    sub_losses = losses[subset_idx]

    valid_k = [k for k in range(max(2, epoch_min), epochs) if not np.isnan(eta[k])]
    delta2L = (sub_losses[:, valid_k] - 2 * sub_losses[:, [k - 1 for k in valid_k]]
               + sub_losses[:, [k - 2 for k in valid_k]])
    gnorm = norms[:, valid_k]

    abs_d2l = np.abs(delta2L).flatten()
    gnorm_flat = gnorm.flatten()
    mask = np.isfinite(abs_d2l) & np.isfinite(gnorm_flat) & (abs_d2l > 0)
    abs_d2l, gnorm_flat = abs_d2l[mask], gnorm_flat[mask]
    sqrt_d2l = np.sqrt(abs_d2l)

    corr = np.corrcoef(sqrt_d2l, gnorm_flat)[0, 1]
    slope = np.polyfit(sqrt_d2l, gnorm_flat, 1)[0]

    delta = CALIBRATED_DELTA[tag]
    passes = abs_d2l < delta
    mean_g_pass = gnorm_flat[passes].mean() if passes.any() else float("nan")
    mean_g_fail = gnorm_flat[~passes].mean() if (~passes).any() else float("nan")
    median_g_all = np.median(gnorm_flat)
    frac_pass_above_median = float(np.mean(gnorm_flat[passes] > median_g_all)) if passes.any() else float("nan")

    print(f"\n=== {tag}: ||g_i|| vs |Delta^2 L_i| ===")
    print(f"n={len(abs_d2l)}, calibrated delta={delta:.3e}, "
          f"{passes.sum()}/{len(passes)} pairs pass the mastery check (|Delta^2L_i|<delta)")
    print(f"corr(||g_i||, sqrt(|Delta^2L_i|)) = {corr:.4f}  (thm:bridge predicts positive)")
    print(f"slope (||g_i|| ~ sqrt(|Delta^2L_i|))  = {slope:.4f}")
    print(f"mean ||g_i|| | passes mastery check  = {mean_g_pass:.4f}")
    print(f"mean ||g_i|| | fails mastery check    = {mean_g_fail:.4f}  "
          f"(ratio pass/fail = {mean_g_pass / mean_g_fail:.3f})")
    print(f"frac(passes AND ||g_i|| > population median {median_g_all:.4f}) = {frac_pass_above_median:.3f}  "
          f"(thm:bridge predicts this should be low, not 0.5)")

    return dict(tag=tag, sqrt_d2l=sqrt_d2l, gnorm=gnorm_flat, corr=corr, slope=slope,
                mean_g_pass=mean_g_pass, mean_g_fail=mean_g_fail,
                frac_pass_above_median=frac_pass_above_median)


def main():
    results = [analyze(tag) for tag in SCHEDULES]
    print("\n\n########## LATE-TRAINING ONLY (epoch >= 150/200, the near-convergence "
          "regime thm:curvature's own text says drift and curvature should both shrink "
          "together) ##########")
    late_results = [analyze(tag, epoch_min=150, label="epoch>=150") for tag in SCHEDULES]

    # Sized/fonted for full-textwidth (figure*) display in the paper, not a single
    # column: previously this rendered at \columnwidth, which shrank a 13in-wide,
    # small-font (7-10pt) figure down to ~3.3in, making the title/tick/axis text
    # illegible. Bigger canvas + explicitly bumped font sizes throughout fix that.
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.3), sharex=False, sharey=False)
    for ax, r in zip(axes, results):
        # subsample for plotting if huge
        n_plot = min(20000, len(r["actual"]))
        idx = np.random.RandomState(0).choice(len(r["actual"]), n_plot, replace=False)
        ax.scatter(r["predicted"][idx], r["actual"][idx], s=4, alpha=0.2, color="#3B76B5")
        lims = [min(r["predicted"].min(), r["actual"].min()), max(r["predicted"].max(), r["actual"].max())]
        ax.plot(lims, lims, color="#CC2936", linestyle="--", linewidth=2, label="y=x")
        ax.set_xlabel(r"predicted: $\eta_t D_i^{(t)}$", fontsize=17)
        ax.set_ylabel(r"actual: $\Delta^2 L_i^{(t)}$", fontsize=17)
        ax.set_title(f"{r['tag']}  (corr={r['corr']:.3f}, slope={r['slope']:.2f})", fontsize=17)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
    fig.suptitle(r"Direct instrumentation of $\Delta^2 L_i^{(t)} = \eta_t D_i^{(t)} + O(\eta_t^2) + R_{i,t}$"
                 " (thm:curvature)", fontsize=19)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_drift_term_validation.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved {out_path}")
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})

    print("\n\n########## ITEM #17: ||g_i^(t)|| vs |Delta^2 L_i^(t)| ##########")
    norm_results = [analyze_norms(tag) for tag in SCHEDULES]
    norm_results = [r for r in norm_results if r is not None]
    if norm_results:
        fig2, axes2 = plt.subplots(1, len(norm_results), figsize=(4.3 * len(norm_results), 4.2))
        if len(norm_results) == 1:
            axes2 = [axes2]
        for ax, r in zip(axes2, norm_results):
            n_plot = min(20000, len(r["gnorm"]))
            idx = np.random.RandomState(0).choice(len(r["gnorm"]), n_plot, replace=False)
            ax.scatter(r["sqrt_d2l"][idx], r["gnorm"][idx], s=2, alpha=0.15)
            ax.set_xlabel(r"$\sqrt{|\Delta^2 L_i^{(t)}|}$")
            ax.set_ylabel(r"$\|g_i^{(t)}\|$")
            ax.set_title(f"{r['tag']} (corr={r['corr']:.3f})")
        fig2.suptitle(r"Item #17: gradient norm vs.\ $\sqrt{|\Delta^2 L_i^{(t)}|}$ (thm:bridge's predicted form)")
        fig2.tight_layout()
        out_path2 = os.path.join(FIG_DIR, "fig_bridge_norm_check.png")
        fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
        fig2.savefig(out_path2.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"\nSaved {out_path2}")


if __name__ == "__main__":
    main()
