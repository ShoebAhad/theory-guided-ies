
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from common import (device, get_cifar10_dataset, get_cifar100_dataset, get_model,
                     get_optimizer_and_scheduler, setup_seed, CifarDataset)
from train import build_transform

SPLIT_SEED = 20260827  # fixed, independent of --seed: same holdout/IL model for every seed
HOLDOUT_FRAC = 0.10
IL_EPOCHS = 40
BATCH_SIZE = 64


class Args:
    momentum = 0.9
    weight_decay = 5e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--root_dir", default=None)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50",
                                                             "resnet101", "vgg16", "densenet121"])
    p.add_argument("--out_dir", default="il_cache")
    p.add_argument("--num_workers", type=int, default=4)
    a = p.parse_args()

    setup_seed(SPLIT_SEED)
    root_dir = a.root_dir or (f"./{'cifar-10' if a.dataset == 'cifar10' else 'cifar-100'}")
    num_classes = 10 if a.dataset == "cifar10" else 100
    get_dataset = get_cifar10_dataset if a.dataset == "cifar10" else get_cifar100_dataset

    train_transform = build_transform(a.dataset)
    # deterministic, non-augmented transform for the final scoring pass (Step 3
    # in the docstring): same normalization, no random crop/flip, so the cached
    # score is a fixed number rather than one random augmentation's outcome.
    if a.dataset == "cifar10":
        score_transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])
    else:
        score_transform = Compose([ToTensor(), Normalize((0.507, 0.487, 0.441),
                                                           (0.267, 0.256, 0.276))])

    train_dataset, _ = get_dataset(root_dir, train_transform)
    N = len(train_dataset)
    data_arr, labels_arr = train_dataset.data, train_dataset.labels

    rng = np.random.RandomState(SPLIT_SEED)
    holdout_size = int(round(HOLDOUT_FRAC * N))
    holdout_idx = np.sort(rng.choice(N, size=holdout_size, replace=False))

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.dataset}_{a.model}"
    il_path = os.path.join(a.out_dir, f"il_losses_{tag}.npy")
    holdout_path = os.path.join(a.out_dir, f"il_holdout_idx_{tag}.npy")
    if os.path.exists(il_path) and os.path.exists(holdout_path):
        print(f"SKIP (already cached): {il_path}")
        return

    holdout_ds = CifarDataset(data_arr, labels_arr, holdout_idx.tolist(), train_transform)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=a.num_workers, drop_last=True)

    model = get_model(a.model, num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler("SGD_exponential_learning_rate",
                                                         model.parameters(), Args())
    ce_none = nn.CrossEntropyLoss(reduction="none")

    print(f"=== pretrain_il_model: {tag}, holdout={holdout_size}/{N} ({HOLDOUT_FRAC:.0%}), "
          f"{IL_EPOCHS} epochs ===", flush=True)
    t0 = time.time()
    il_cum_backprop = 0
    model.train()
    for epoch in range(IL_EPOCHS):
        te = time.time()
        for inputs, labels, _ in holdout_loader:
            if inputs.size(0) == 1:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ce_none(outputs, labels).mean()
            loss.backward()
            optimizer.step()
            il_cum_backprop += inputs.size(0)
        scheduler.step()
        print(f"[il_pretrain:{tag}] epoch {epoch+1}/{IL_EPOCHS} time={time.time()-te:.1f}s", flush=True)

    # Step 3: frozen forward pass over the WHOLE training set (holdout entries'
    # cached losses are never used by train.py since rho_loss permanently
    # excludes holdout_mask, but computing them uniformly keeps this script simple).
    model.eval()
    full_ds = CifarDataset(data_arr, labels_arr, list(range(N)), score_transform)
    full_loader = DataLoader(full_ds, batch_size=256, shuffle=False, num_workers=a.num_workers)
    il_losses = np.full(N, np.nan, dtype=np.float32)
    with torch.no_grad():
        for inputs, labels, indices in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses = ce_none(outputs, labels)
            il_losses[indices.numpy()] = losses.detach().float().cpu().numpy()
    assert not np.isnan(il_losses).any(), "every instance must get a cached IL loss"

    np.save(il_path, il_losses)
    np.save(holdout_path, holdout_idx)
    total_time = time.time() - t0
    print(f"DONE {tag}: saved {il_path} ({il_losses.shape}), {holdout_path} ({holdout_idx.shape}) "
          f"pretrain_backprop_instances={il_cum_backprop} pretrain_time_s={total_time:.1f} "
          f"(add this backprop/time cost on top of each seed's main-run cum_backprop when "
          f"reporting rho_loss's total compute -- it is shared/amortized across all seeds "
          f"of this dataset/model pair, not per-seed)", flush=True)


if __name__ == "__main__":
    main()
