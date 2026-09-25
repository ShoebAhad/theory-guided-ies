
import csv
import glob
import os

import torch
from torch.utils.flop_counter import FlopCounterMode

from common import get_model, device

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARCHS = ["resnet18", "vgg16", "densenet121", "vit_tiny_cifar"]
BATCH = 64


def measure_flops(model_name, num_classes=10):
    model = get_model(model_name, num_classes).to(device)
    model.train()
    x = torch.randn(BATCH, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (BATCH,), device=device)
    ce = torch.nn.CrossEntropyLoss()

    # forward-only FLOPs
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(model, display=False) as fc_fwd:
        out = model(x)
        _ = ce(out, y)  # loss forward is negligible but included for consistency
    fwd_flops = fc_fwd.get_total_flops()

    # forward+backward FLOPs
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(model, display=False) as fc_full:
        out = model(x)
        loss = ce(out, y)
        loss.backward()
    full_flops = fc_full.get_total_flops()

    bwd_flops = full_flops - fwd_flops
    return fwd_flops / BATCH, bwd_flops / BATCH  # per-instance


def per_run_flop_saved(csv_path, f_fwd, f_bwd):
    """Every epoch: N instances forwarded (active+passive), backprop_this_epoch
    instances backprop'd. Baseline forwards+backprops all N every epoch."""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    epochs = len(rows)
    n_active0 = int(rows[0]["active_set_size"])  # baseline: active_set_size==N always
    # N = active_set_size when nothing has ever been excluded (epoch 1) for any method,
    # since epoch 1 always has the full active set before any mastery check can fire.
    N = int(rows[0]["active_set_size"])
    cum_backprop = int(rows[-1]["cum_backprop_instances"])

    flops_used = epochs * N * f_fwd + cum_backprop * f_bwd
    flops_baseline = epochs * N * (f_fwd + f_bwd)
    flop_saved_pct = 100.0 * (1 - flops_used / flops_baseline)
    backprop_saved_pct = 100.0 * (1 - cum_backprop / (N * epochs))
    return flop_saved_pct, backprop_saved_pct, N, epochs, cum_backprop


def main():
    print("Measuring per-instance FLOPs (batch=64, 32x32x3 input)...")
    flop_stats = {}
    for arch in ARCHS:
        f_fwd, f_bwd = measure_flops(arch, num_classes=10)
        ratio = f_bwd / f_fwd
        flop_stats[arch] = (f_fwd, f_bwd)
        print(f"  {arch}: fwd={f_fwd/1e6:.2f} MFLOPs/inst  bwd={f_bwd/1e6:.2f} MFLOPs/inst  "
              f"bwd/fwd={ratio:.3f}")

    out_path = os.path.join(RESULTS_DIR, "flop_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "model", "fwd_mflops_per_inst", "bwd_mflops_per_inst",
                     "bwd_fwd_ratio", "N", "epochs", "cum_backprop_instances",
                     "backprop_saved_pct", "flop_saved_pct"])
        for csv_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))):
            name = os.path.basename(csv_path)[:-4]
            if name in ("summary", "flop_summary"):
                continue
            arch = next((a for a in ARCHS if f"_{a}_" in name), None)
            if arch is None:
                continue
            f_fwd, f_bwd = flop_stats[arch]
            result = per_run_flop_saved(csv_path, f_fwd, f_bwd)
            if result is None:
                continue
            flop_saved_pct, backprop_saved_pct, N, epochs, cum_backprop = result
            w.writerow([name, arch, f_fwd/1e6, f_bwd/1e6, f_bwd/f_fwd, N, epochs,
                        cum_backprop, backprop_saved_pct, flop_saved_pct])
            print(f"{name}: backprop_saved={backprop_saved_pct:.2f}%  "
                  f"flop_saved={flop_saved_pct:.2f}%")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
