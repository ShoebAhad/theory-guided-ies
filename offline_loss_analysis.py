"""CPU-only offline analyses, run on the saved full-training
per-instance loss matrices results/drift_{exponential,fixed,Adam}_losses.npy
(N=50000 x 200 epochs, ResNet-18/CIFAR-10, baseline training, seed 0).

  A1  noise-scale estimators vs epoch (spec + bias of the shipped estimator)
  A2  calibration epoch t_c: delta(t_c) and an open-loop replay of the
      exclusion rule on the baseline trajectory
  A3  order of the difference without selection bias: pre-registered epochs,
      signal-independent targets, matched flag rate, N in {0,1,2,3}
  A4  nuisance structure the orders differ on (loss plateaus, linear drift)
  A5  patience: per-instance / aggregate / error-control K values
  A6  synthetic check of the detection-power ordering (Prop. in App. R)

Caveat carried into every table: these losses were recorded in eval mode at the
end-of-epoch parameters (drift_term_run.py), whereas train.py's mastery check uses
train-mode forward losses; absolute sigma values therefore differ from the logged
calibration, and the replay in A2 is open-loop (no feedback from exclusion).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rankdata

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_DIR = os.path.join(RESULTS_DIR, "offline")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "offline")
SCHEDS = ["exponential", "fixed", "Adam"]
NOISE_MULT = {0: 1.0, 1: 2.0, 2: 6.0, 3: 20.0}
PHI_INV_0625 = norm.ppf(0.625)
P_BUDGET, KSTAR = 0.3, 15


def k_gauss(kstar=KSTAR, p_budget=P_BUDGET):
    p_check = 1.0 - (1.0 - p_budget) ** (1.0 / kstar)
    return norm.ppf(1 - p_check / 2)


def diff(L, order, t):
    """|D^order L| at column t (order 0 = the loss itself)."""
    coeffs = {0: (1,), 1: (1, -1), 2: (1, -2, 1), 3: (1, -3, 3, -1)}[order]
    out = np.zeros(L.shape[0], dtype=np.float64)
    for j, c in enumerate(coeffs):
        out += c * L[:, t - j].astype(np.float64)
    return np.abs(out)


def sigma_lowerhalf(a, mult):
    lower = a[a <= np.median(a)]
    return math.sqrt(max(np.median(lower ** 2) / mult, 1e-24))


def sigma_q25(a, mult):
    return np.percentile(a, 25.0) / PHI_INV_0625 / math.sqrt(mult)


def sigma_temporal(window):
    W = window.shape[1]
    t = np.arange(W, dtype=np.float64)
    X = np.stack([np.ones(W), t, t ** 2], axis=1)
    Y = window.T.astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    dof = W - 3
    s2 = ((Y - X @ beta) ** 2).sum(axis=0) / dof
    chi2_med = dof * (1 - 2.0 / (9 * dof)) ** 3
    return math.sqrt(np.median(s2) * dof / chi2_med)


def auroc(score_pos_high, y):
    """AUROC of `score_pos_high` (higher = predicted positive) for binary y."""
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score_pos_high)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(s):
    L = np.load(os.path.join(RESULTS_DIR, f"drift_{s}_losses.npy"))
    eta = np.load(os.path.join(RESULTS_DIR, f"drift_{s}_eta.npy"))
    return L, eta


# ----------------------------------------------------------------------------- A1
def a1_sigma(all_L, summary):
    rng = np.random.default_rng(0)
    mc = {}
    z = rng.normal(size=(200000, 3))
    d2 = np.abs(z[:, 0] - 2 * z[:, 1] + z[:, 2])
    mc["pure_noise_sigma1"] = {"lowerhalf": sigma_lowerhalf(d2, 6), "q25": sigma_q25(d2, 6),
                               "temporal": sigma_temporal(rng.normal(size=(20000, 10)))}
    # noise + a geometric trend of comparable size in half of the instances
    t = np.arange(10)
    trend = (rng.random((20000, 1)) < 0.5) * 3.0 * 0.8 ** t[None, :]
    Y = trend + rng.normal(size=(20000, 10))
    d2m = np.abs(Y[:, -1] - 2 * Y[:, -2] + Y[:, -3])
    mc["half_trend_sigma1"] = {"lowerhalf": sigma_lowerhalf(d2m, 6), "q25": sigma_q25(d2m, 6),
                               "temporal": sigma_temporal(Y)}
    summary["A1_monte_carlo"] = mc

    rows = {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, s in zip(axes, SCHEDS):
        L = all_L[s]
        T = L.shape[1]
        ep, lh, q, tm = [], [], [], []
        for c in range(9, T):
            a = diff(L, 2, c)
            ep.append(c + 1)
            lh.append(sigma_lowerhalf(a, 6))
            q.append(sigma_q25(a, 6))
            tm.append(sigma_temporal(L[:, c - 9:c + 1]))
        rows[s] = {"epoch": ep, "lowerhalf": lh, "q25": q, "temporal": tm}
        ax.semilogy(ep, lh, label="lower-half (shipped)")
        ax.semilogy(ep, q, label="q25 (consistent)")
        ax.semilogy(ep, tm, label="temporal W=10")
        ax.axvline(31, color="k", ls=":", lw=0.8)
        ax.set_title(s)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel(r"$\hat\sigma$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_r_sigma_estimators.png"), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "fig_r_sigma_estimators.pdf"))
    plt.close(fig)
    with open(os.path.join(OUT_DIR, "A1_sigma_by_epoch.json"), "w") as f:
        json.dump(rows, f)
    at31 = {s: {k: rows[s][k][rows[s]["epoch"].index(31)] for k in ("lowerhalf", "q25", "temporal")}
            for s in SCHEDS}
    summary["A1_sigma_at_epoch31"] = at31
    return rows


# ----------------------------------------------------------------------------- A2
def replay(L, t_c, delta, kstar, precalib="none", pre_delta=1e-3):
    """Open-loop replay of the TG-IES counting rule on a fixed trajectory.
    Returns per-epoch passive fraction (passive at the START of each epoch)."""
    N, T = L.shape
    active = np.ones(N, dtype=bool)
    cnt = np.zeros(N, dtype=np.int64)
    passive_frac = np.zeros(T)
    for c in range(T):
        passive_frac[c] = 1 - active.mean()
        if c < 2:
            continue
        e = c  # 0-indexed epoch; code calibrates at e == t_c
        if e < t_c and precalib == "none":
            continue
        d, k = (pre_delta, 1) if e < t_c else (delta, kstar)
        if e == t_c and precalib == "none":
            cnt[:] = 0
        below = diff(L, 2, c) < d
        cnt[below] += 1
        cnt[~below] = 0
        active[~below] = True
        active[below & (cnt >= k)] = False
    return passive_frac


def a2_tc(all_L, all_eta, summary):
    kg = k_gauss()
    out = {}
    tcs = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
    for s in SCHEDS:
        L = all_L[s]
        res = []
        for tc in tcs:
            a = diff(L, 2, tc)  # losses observed at 0-indexed epoch tc
            sig_lh, sig_q = sigma_lowerhalf(a, 6), sigma_q25(a, 6)
            d_lh, d_q = kg * math.sqrt(6) * sig_lh, kg * math.sqrt(6) * sig_q
            pf = replay(L, tc, d_lh, KSTAR)
            pf_q = replay(L, tc, d_q, KSTAR)
            res.append({"t_c": tc, "lr_at_tc": float(all_eta[s][tc]), "sigma_lowerhalf": sig_lh,
                        "delta_lowerhalf": d_lh, "delta_q25": d_q,
                        "openloop_saved_pct_lowerhalf": 100 * pf.mean(),
                        "openloop_saved_pct_q25": 100 * pf_q.mean(),
                        "final_passive_pct_lowerhalf": 100 * pf[-1]})
        # the shipped behavior (IES warm-up before t_c=30) for reference
        a = diff(L, 2, 30)
        pf_ies = replay(L, 30, kg * math.sqrt(6) * sigma_lowerhalf(a, 6), KSTAR, precalib="ies")
        out[s] = {"grid": res, "openloop_saved_pct_tc30_precalib_ies": 100 * pf_ies.mean()}
    summary["A2_tc"] = out

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for s in SCHEDS:
        g = out[s]["grid"]
        axes[0].semilogy([r["t_c"] for r in g], [r["delta_lowerhalf"] for r in g], "o-", label=s)
        axes[1].plot([r["t_c"] for r in g], [r["openloop_saved_pct_lowerhalf"] for r in g], "o-", label=s)
    axes[0].set_xlabel("calibration epoch $t_c$"); axes[0].set_ylabel(r"calibrated $\delta$")
    axes[1].set_xlabel("calibration epoch $t_c$"); axes[1].set_ylabel("open-loop saved (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_r_tc_openloop.png"), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "fig_r_tc_openloop.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- A3
def a3_order(all_L, summary, H=20):
    kg = k_gauss()
    eval_cols = list(range(39, 180, 20))  # epochs 40, 60, ..., 180 (fixed in advance)
    qs = [0.1, 0.2, 0.3, 0.5, 0.7]
    out = {}
    for s in SCHEDS:
        L = all_L[s].astype(np.float64)
        per = []
        for c in eval_cols:
            sig = sigma_q25(diff(L, 2, c), 6)
            fut = L[:, c + 1:c + 1 + H]
            # targets use only FUTURE epochs (disjoint from the t-3..t window the signals use)
            remaining = L[:, c] - fut.min(axis=1)          # learnable progress still ahead
            frange = fut.max(axis=1) - fut.min(axis=1)     # future stability
            eps = 3 * sig
            targets = {"converged": remaining <= eps, "stable": frange <= 2 * eps}
            row = {"epoch": c + 1, "sigma": sig}
            for name, y in targets.items():
                row[f"base_rate_{name}"] = float(y.mean())
                for n in (0, 1, 2, 3):
                    S = diff(L, n, c)
                    row[f"auroc_{name}_N{n}"] = auroc(-S, y)
                    order = np.argsort(S, kind="stable")
                    for q in qs:
                        flagged = order[: int(q * len(S))]
                        row[f"prec_{name}_N{n}_q{q}"] = float(y[flagged].mean())
            # noise-calibrated rule at the SAME nominal per-check false-retention for every N
            for n in (1, 2, 3):
                S = diff(L, n, c)
                flagged = S < kg * math.sqrt(NOISE_MULT[n]) * sig
                row[f"calib_flag_frac_N{n}"] = float(flagged.mean())
                row[f"calib_prec_converged_N{n}"] = float(targets["converged"][flagged].mean()) if flagged.any() else float("nan")
            per.append(row)
        out[s] = per

        # the paper's original protocol, evaluated at EACH order's own first-crossing epoch
        sig31 = sigma_lowerhalf(diff(L, 2, 30), 6)
        cross = {}
        for n_sel in (1, 2, 3):
            thr = kg * math.sqrt(NOISE_MULT[n_sel]) * sig31
            first = np.full(L.shape[0], -1)
            for c in range(30, L.shape[1]):
                hit = (first < 0) & (diff(L, n_sel, c) < thr)
                first[hit] = c
            ok = first >= 0
            idx = np.nonzero(ok)[0]
            mse = {}
            for n in (1, 2, 3):
                coeffs = {1: (1, -1), 2: (1, -2, 1), 3: (1, -3, 3, -1)}[n]
                vals = sum(cf * L[idx, first[idx] - j] for j, cf in enumerate(coeffs))
                mse[f"N{n}"] = float(np.mean(vals ** 2))
            cross[f"selected_by_N{n_sel}"] = {"n_instances": int(ok.sum()), "mse": mse,
                                              "argmin": min(mse, key=mse.get)}
        out[s + "_first_crossing"] = cross
    summary["A3_order"] = out

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, s in zip(axes, SCHEDS):
        per = out[s]
        for n in (0, 1, 2, 3):
            ax.plot([r["epoch"] for r in per], [r[f"auroc_converged_N{n}"] for r in per], "o-",
                    label=f"N={n}" + (" (loss)" if n == 0 else ""))
        ax.set_title(s); ax.set_xlabel("pre-registered epoch"); ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("AUROC (target: converged)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_r_order_auroc.png"), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "fig_r_order_auroc.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- A4
def a4_nuisance(all_L, summary):
    out = {}
    for s in SCHEDS:
        L = all_L[s].astype(np.float64)
        sig = sigma_q25(diff(L, 2, 199), 6)
        plateau = np.median(L[:, -20:], axis=1)
        W = L[:, 120:200]
        t = np.arange(W.shape[1], dtype=np.float64)
        tc = t - t.mean()
        slope = (W * tc).sum(axis=1) / (tc ** 2).sum()
        resid = W - W.mean(axis=1, keepdims=True) - slope[:, None] * tc
        se = np.sqrt((resid ** 2).sum(axis=1) / (len(t) - 2) / (tc ** 2).sum())
        z = slope / np.maximum(se, 1e-30)
        out[s] = {"sigma_final": sig,
                  "plateau_quantiles": {str(q): float(np.quantile(plateau, q)) for q in (0.5, 0.9, 0.99)},
                  "frac_plateau_gt_3sigma": float((plateau > 3 * sig).mean()),
                  "frac_plateau_gt_delta": float((plateau > k_gauss() * math.sqrt(6) * sig).mean()),
                  "frac_significant_linear_drift_|z|>3": float((np.abs(z) > 3).mean()),
                  "frac_positive_drift_z>3": float((z > 3).mean()),
                  "median_|slope|_over_sigma_per_epoch": float(np.median(np.abs(slope)) / sig)}
    summary["A4_nuisance"] = out


# ----------------------------------------------------------------------------- A5
def a5_kstar(summary):
    N, tau, beta = 50000, 0.02, 1.0
    eta31 = 0.1 * 0.96 ** 30
    eps0 = 0.02954  # mean calibrated eps0, primary cell, 3 seeds (reactivation probe logs)
    out = {"eta_at_calibration": eta31, "eps0": eps0,
           "K_per_instance": math.floor(N * tau / (beta * eta31 * eps0)),
           "K_aggregate_worstcase_P=N": math.floor(tau / (beta * eta31 * eps0))}
    for pf in (0.5, 0.875):
        out[f"K_aggregate_P={pf}N"] = math.floor(tau / (beta * eta31 * pf * eps0))
    # error-control patience: P(K passes | margin x*sqrt6*sigma beyond delta) <= (2e^{-x^2/2})^{ceil(K/3)}
    err = {}
    for alpha in (1e-2, 1e-3, 1e-4):
        for x in (1.5, 2.0, 2.5, 3.0):
            per_block = 2 * math.exp(-x * x / 2)
            if per_block >= 1:
                continue
            B = math.ceil(math.log(alpha) / math.log(per_block))
            err[f"alpha={alpha:g},x={x}"] = 3 * B - 2
    out["K_error_control"] = err
    summary["A5_kstar"] = out


# ----------------------------------------------------------------------------- A6
def a6_synthetic(summary, n=200000, seed=0):
    """Instances at a fixed epoch t with excess e=c r^t, plateau l_i and drift a_i.
    For each order N, flag with the noise-calibrated threshold k*sqrt(C(2N,N))*sigma
    and report erroneous exclusion (flagged although e>eps) / erroneous retention."""
    rng = np.random.default_rng(seed)
    sigma, r, t = 1.0, 0.8, 30
    kg = k_gauss()
    out = {}
    for nuis in ("none", "plateau", "plateau+drift"):
        e = rng.exponential(4.0, n)
        c = e / r ** t
        ell = rng.exponential(5.0, n) if nuis != "none" else np.zeros(n)
        a = rng.normal(0, 1.0, n) if nuis == "plateau+drift" else np.zeros(n)
        tt = np.arange(t - 3, t + 1)
        Ltraj = ell[:, None] + a[:, None] * tt[None, :] + c[:, None] * r ** tt[None, :] \
            + rng.normal(0, sigma, (n, 4))
        Lt = Ltraj[:, ::-1]  # column j = L^{t-j}
        mastered = e <= 1.0
        res = {}
        for N in (0, 1, 2, 3):
            S = diff(Lt, N, 0) if N > 0 else np.abs(Lt[:, 0])
            flagged = S < kg * math.sqrt(NOISE_MULT[N]) * sigma
            res[f"N{N}"] = {"flag_rate": float(flagged.mean()),
                            "erroneous_exclusion": float((flagged & ~mastered).sum() / max(flagged.sum(), 1)),
                            "erroneous_retention": float((~flagged & mastered).sum() / max(mastered.sum(), 1)),
                            "auroc": auroc(-S, mastered)}
        out[nuis] = res
    summary["A6_synthetic"] = out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    all_L, all_eta = {}, {}
    for s in SCHEDS:
        all_L[s], all_eta[s] = load(s)
    summary = {}
    a1_sigma(all_L, summary)
    a2_tc(all_L, all_eta, summary)
    a3_order(all_L, summary)
    a4_nuisance(all_L, summary)
    a5_kstar(summary)
    a6_synthetic(summary)
    with open(os.path.join(OUT_DIR, "offline_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:20000])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CPU-only offline analyses on saved per-instance loss matrices")
    ap.add_argument("--out_dir", default=OUT_DIR, help="where JSON summaries are written")
    ap.add_argument("--fig_dir", default=FIG_DIR, help="where figures are written")
    _a = ap.parse_args()
    OUT_DIR, FIG_DIR = _a.out_dir, _a.fig_dir
    main()
