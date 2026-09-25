
import csv
import os
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

LABELS = {
    ("cifar10", "resnet18", "SGD_exponential_learning_rate"): "CIFAR-10 / ResNet-18 / SGD-exp (primary)",
    ("cifar10", "resnet18", "SGD_fixed_learning_rate"): "CIFAR-10 / ResNet-18 / SGD-fixed",
    ("cifar10", "resnet18", "SGD_linear_learning_rate"): "CIFAR-10 / ResNet-18 / SGD-linear",
    ("cifar10", "resnet18", "SGD_step_learning_rate"): "CIFAR-10 / ResNet-18 / SGD-step",
    ("cifar10", "resnet18", "SGD_cosine_learning_rate"): "CIFAR-10 / ResNet-18 / SGD-cosine",
    ("cifar10", "resnet18", "Adam"): "CIFAR-10 / ResNet-18 / Adam",
    ("cifar10", "vgg16", "SGD_exponential_learning_rate"): "CIFAR-10 / VGG-16 / SGD-exp",
    ("cifar10", "densenet121", "SGD_exponential_learning_rate"): "CIFAR-10 / DenseNet-121 / SGD-exp",
    ("cifar100", "resnet18", "SGD_exponential_learning_rate"): "CIFAR-100 / ResNet-18 / SGD-exp",
    ("cifar10", "vit_tiny_cifar", "AdamW_warmup_cosine"): "CIFAR-10 / ViT-Tiny / AdamW-warmup-cosine",
    ("cifar100", "vit_tiny_cifar", "AdamW_warmup_cosine"): "CIFAR-100 / ViT-Tiny / AdamW-warmup-cosine",
}


def mean_std(xs):
    if len(xs) == 1:
        return xs[0], None
    return statistics.mean(xs), statistics.stdev(xs)


def main():
    rows = {}
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            if row["method"] != "tgies":
                continue
            # summary.csv is train.py's flat append-log of EVERY run, including one-off
            # ablation/sensitivity-sweep variants that reuse seed=0 with a suffixed
            # run_name (e.g. "..._seed0_epspct20", "..._seed0_ablation_deltaonly").
            # Only the canonical "{method}_{dataset}_{model}_{optimizer}_seed{seed}" name
            # (no suffix) is one of this table's real per-cell runs.
            canonical = f"{row['method']}_{row['dataset']}_{row['model']}_{row['optimizer']}_seed{row['seed']}"
            if row["run_name"] != canonical:
                continue
            key = (row["dataset"], row["model"], row["optimizer"])
            rows.setdefault(key, []).append(row)

    print(f"{'Cell':45s} {'n':>2s}  {'Acc (%)':>14s}  {'Saved (%)':>14s}")
    missing = [k for k in LABELS if k not in rows]
    for key, label in LABELS.items():
        entries = rows.get(key, [])
        seeds = sorted(set(int(r["seed"]) for r in entries))
        n = len(seeds)
        if n == 0:
            print(f"{label:45s}  0  MISSING")
            continue
        accs = [float(r["best_test_acc_pct"]) for r in entries]
        saved = [float(r["backprop_saved_pct"]) for r in entries]
        acc_m, acc_s = mean_std(accs)
        sav_m, sav_s = mean_std(saved)
        acc_str = f"{acc_m:.2f}" + (f"$\\pm${acc_s:.2f}" if acc_s is not None else "*")
        sav_str = f"{sav_m:.2f}" + (f"$\\pm${sav_s:.2f}" if sav_s is not None else "*")
        print(f"{label:45s} {n:>2d}  {acc_str:>14s}  {sav_str:>14s}")

    if missing:
        print(f"\n{len(missing)} cell(s) not yet in summary.csv for method=tgies: {missing}")
    n_seeds_per_cell = {k: len(set(int(r['seed']) for r in v)) for k, v in rows.items()}
    incomplete = [k for k in LABELS if n_seeds_per_cell.get(k, 0) < 3]
    if incomplete:
        print(f"\n{len(incomplete)} cell(s) still below 3 seeds (marked '*' above): {incomplete}")
        print("Re-run once item 1's backfills land for a fully seed-averaged table.")


if __name__ == "__main__":
    main()
