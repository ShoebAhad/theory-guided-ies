
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from common import device, get_cifar10_dataset, get_model, get_optimizer_and_scheduler, setup_seed

SEED = 0
EPOCHS = 200
DUMP_FROM = 100  # only dump epochs >= this (late-training regime IES/TG-IES targets)
BATCH_SIZE = 64
OUT_DIR = "results"
DUMP_PATH = os.path.join(OUT_DIR, "noise_autocorr_losses.npy")


class Args:
    momentum = 0.9
    weight_decay = 5e-4


def build_transform():
    return Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
                     Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def main():
    setup_seed(SEED)
    transform = build_transform()
    train_dataset, _ = get_cifar10_dataset("./cifar-10", transform)
    N = len(train_dataset)

    model = get_model("resnet18", 10).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler("SGD_exponential_learning_rate",
                                                         model.parameters(), Args())
    ce_none = nn.CrossEntropyLoss(reduction="none")

    os.makedirs(OUT_DIR, exist_ok=True)
    n_dump = EPOCHS - DUMP_FROM
    dump = np.lib.format.open_memmap(DUMP_PATH, mode="w+", dtype=np.float32, shape=(N, n_dump))
    dump[:] = np.nan

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, drop_last=True)
        epoch_losses = np.full(N, np.nan, dtype=np.float32) if epoch >= DUMP_FROM else None

        for inputs, labels, indices in loader:
            if inputs.size(0) == 1:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            idx_np = indices.numpy()
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = ce_none(outputs, labels)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            if epoch_losses is not None:
                epoch_losses[idx_np] = losses.detach().float().cpu().numpy()

        scheduler.step()
        if epoch >= DUMP_FROM:
            dump[:, epoch - DUMP_FROM] = epoch_losses
            dump.flush()

        print(f"[noise_autocorr] epoch {epoch+1}/{EPOCHS} lr={optimizer.param_groups[0]['lr']:.5f} "
              f"time={time.time()-t0:.1f}s", flush=True)

    print(f"DONE. Dumped losses for epochs {DUMP_FROM+1}..{EPOCHS} to {DUMP_PATH} "
          f"(shape={dump.shape})")


if __name__ == "__main__":
    main()
