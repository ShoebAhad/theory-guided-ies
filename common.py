
import math
import os
import random
import _pickle as cPickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from resnet import ResNet18, ResNet34, ResNet50, ResNet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def unpickle(file):
    with open(file, 'rb') as fo:
        d = cPickle.load(fo, encoding='bytes')
    return d


class CifarDataset(Dataset):
    def __init__(self, data, labels, index_list, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        true_idx = self.index_list[idx]
        img, target = self.data[true_idx], self.labels[true_idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target, true_idx


class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._make_layer(3, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 3),
            self._make_layer(256, 512, 3),
            self._make_layer(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x, ret_feat=False):
        x = self.features(x)
        feat = x.view(x.size(0), -1)
        out = self.classifier(feat)
        if ret_feat:
            return out, feat
        return out


class DenseNet121_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, ret_feat=False):
        feat = self.model.features(x)
        feat = nn.functional.relu(feat, inplace=True)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        out = self.model.classifier(feat)
        if ret_feat:
            return out, feat
        return out


class ViT_CIFAR(nn.Module):
    """CIFAR-native ViT-Tiny: patch-4 on 32x32 input, 6 layers, 192-dim, 3 heads.

    Deliberately NOT torchvision/timm's `vit_tiny_patch16_224`: at patch-16 a
    32x32 image yields only 4 tokens, and those configs assume ImageNet-scale
    pretraining. Patch-4 gives 8x8=64 tokens, which is what makes attention
    meaningful at CIFAR resolution when training from scratch.

    Pre-norm blocks (the standard modern arrangement -- more stable from scratch
    than post-norm). `forward(x, ret_feat=True)` returns the CLS token after the
    final LayerNorm, i.e. exactly the penultimate representation feeding the
    classifier head, matching the (out, feat) contract that ResNet/VGG/DenseNet
    expose here and that TG-IES's gradient-norm proxy relies on.
    """

    def __init__(self, num_classes=10, img_size=32, patch_size=4, dim=192,
                 depth=6, heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        n_patches = (img_size // patch_size) ** 2

        # patch embedding as a strided conv == linear projection of flat patches
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(dim, heads, int(dim * mlp_ratio), dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, ret_feat=False):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)       # (B, n_patches, dim)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        feat = self.norm(x)[:, 0]                                 # CLS token
        out = self.head(feat)
        if ret_feat:
            return out, feat
        return out


class ViTBlock(nn.Module):
    """Pre-norm transformer block: x + attn(LN(x)), then x + mlp(LN(x))."""

    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def get_cifar10_dataset(root_dir, transform):
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(root_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']
    train_data = np.concatenate(train_data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    train_labels = np.array(train_labels)

    test_batch = unpickle(os.path.join(root_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    test_labels = np.array(test_batch[b'labels'])

    train_dataset = CifarDataset(train_data, train_labels, list(range(len(train_data))), transform)
    test_dataset = CifarDataset(test_data, test_labels, list(range(len(test_data))), transform)
    return train_dataset, test_dataset


def get_cifar100_dataset(root_dir, transform):
    batch = unpickle(os.path.join(root_dir, 'train'))
    train_data = batch[b'data']
    train_labels = batch[b'fine_labels']
    train_data = train_data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    train_labels = np.array(train_labels)

    test_batch = unpickle(os.path.join(root_dir, 'test'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    test_labels = np.array(test_batch[b'fine_labels'])

    train_dataset = CifarDataset(train_data, train_labels, list(range(len(train_data))), transform)
    test_dataset = CifarDataset(test_data, test_labels, list(range(len(test_data))), transform)
    return train_dataset, test_dataset


def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return ResNet18(num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes)
    elif model_name == "resnet50":
        return ResNet50(num_classes)
    elif model_name == "resnet101":
        return ResNet101(num_classes)
    elif model_name == "vgg16":
        return VGG16_CIFAR(num_classes)
    elif model_name == "densenet121":
        return DenseNet121_CIFAR(num_classes)
    elif model_name == "vit_tiny_cifar":
        return ViT_CIFAR(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_optimizer_and_scheduler(optimizer_name, model_parameters, args):
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    from torch.optim import Adam, AdamW

    if optimizer_name == "Adam":
        optimizer = Adam(model_parameters, lr=0.001)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "Adam_W":
        optimizer = AdamW(model_parameters, lr=0.001, weight_decay=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "SGD_fixed_learning_rate":
        optimizer = optim.SGD(model_parameters, lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "SGD_linear_learning_rate":
        optimizer = optim.SGD(model_parameters, lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=150)
    elif optimizer_name == "SGD_exponential_learning_rate":
        optimizer = optim.SGD(model_parameters, lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    elif optimizer_name == "SGD_step_learning_rate":
        # standard CIFAR step-decay recipe: 10x drops at 30/60/90% of training (60/120/180 of 200 epochs)
        optimizer = optim.SGD(model_parameters, lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
    elif optimizer_name == "SGD_cosine_learning_rate":
        optimizer = optim.SGD(model_parameters, lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0)
    elif optimizer_name == "AdamW_warmup_cosine":
        # Standard from-scratch ViT recipe: AdamW with decoupled wd=0.05, a short
        # linear warmup (transformers diverge from scratch without it) then cosine
        # decay to zero. train.py calls scheduler.step() once per EPOCH, so this
        # LambdaLR is indexed in epochs, not steps.
        warmup_epochs = getattr(args, "warmup_epochs", 10)
        total_epochs = max(args.epochs, warmup_epochs + 1)
        optimizer = AdamW(model_parameters, lr=1e-3, weight_decay=0.05)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer, scheduler
