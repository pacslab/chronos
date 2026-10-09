import os
import re
import math
import time
import json
import random
import argparse
import platform
import warnings
from contextlib import nullcontext
from typing import Tuple, Callable, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42


try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root
    from torch.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = True

def _amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if _USE_TORCH_AMP_ROOT:
        return _autocast_root(device_type='cuda', dtype=torch.float16)
    else:
        return _autocast_cuda(dtype=torch.float16)


FIXED = {
    "adamw": {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.05},
    "sgd":   {"momentum": 0.9, "nesterov": False, "weight_decay": 5e-4},
    "dropout_p": 0.0,
    "activation": "gelu",
}


_TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
_TINY_ZIP = "tiny-imagenet-200.zip"
_TINY_DIR = "tiny-imagenet-200"

def _download_tiny_imagenet(root: str) -> str:
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, _TINY_DIR)
    if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, "train")):
        return target_dir
    zip_path = os.path.join(root, _TINY_ZIP)
    if not os.path.exists(zip_path):
        print("Downloading TinyImageNet-200 (~236MB)...")
        import urllib.request
        urllib.request.urlretrieve(_TINY_URL, zip_path)
    print("Extracting TinyImageNet-200...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return target_dir

class TinyImageNet200(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Callable = None):
        assert split in ("train", "val")
        self.root = _download_tiny_imagenet(root)
        self.split = split
        self.transform = transform

        with open(os.path.join(self.root, "wnids.txt"), "r") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []

        if split == "train":
            train_dir = os.path.join(self.root, "train")
            for wnid in self.wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        else:
            val_dir = os.path.join(self.root, "val")
            ann_path = os.path.join(val_dir, "val_annotations.txt")
            mapping = {}
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
            img_dir = os.path.join(val_dir, "images")
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    wnid = mapping.get(fname, None)
                    if wnid is not None:
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        self.num_classes = 200

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


def tmf_fix(tfm):
    return tfm

def build_cifar100(data_root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tmf_fix(tfm))
    return ds, 100, (3, 32, 32), 'CIFAR100', len(ds)

def build_tiny_imagenet_train(data_root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    ds = TinyImageNet200(root=data_root, split="train", transform=tmf_fix(tfm))
    return ds, 200, (3, 64, 64), 'TinyImageNet', len(ds)

def build_stl10_train(data_root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))
    ])
    ds = datasets.STL10(root=data_root, split='train', download=True, transform=tmf_fix(tfm))
    return ds, 10, (3, 96, 96), 'STL10', len(ds)

DATASET_BUILDERS = {
    'CIFAR100': build_cifar100,
    'TinyImageNet': build_tiny_imagenet_train,
    'STL10': build_stl10_train,
}
IMAGE_DATASETS = set(DATASET_BUILDERS.keys())


ALLOWED_DATASETS_BY_MODEL = {
    'mixer':   ['TinyImageNet', 'STL10'],
    'resmlp':  ['CIFAR100', 'TinyImageNet'],
    'asmlp':   ['TinyImageNet', 'STL10'],
}


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = in_chans * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        x = self.unfold(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_mlp_dim, channel_mlp_dim, p=0.0, act='gelu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU() if act == 'gelu' else nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(token_mlp_dim, num_patches),
            nn.Dropout(p),
        )
        self.ln2 = nn.LayerNorm(dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            the_act,
            nn.Dropout(p),
            nn.Linear(channel_mlp_dim, dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        y = self.ln1(x)
        y = self.token_mlp(y.transpose(1, 2)).transpose(1, 2)
        x = x + y
        y = self.channel_mlp(self.ln2(x))
        return x + y

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=8, token_mlp_dim=128, channel_mlp_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        P = self.patch.num_patches
        self.blocks = nn.Sequential(*[
            MixerBlock(P, embed_dim, token_mlp_dim, channel_mlp_dim, p=p, act=act) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.ln(x).mean(dim=1)
        return self.head(x)

class ResMLPBlock(nn.Module):
    def __init__(self, num_patches, dim, hidden_channel_dim, p=0.0, act='gelu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_linear = nn.Linear(num_patches, num_patches)
        self.drop1 = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_channel_dim),
            the_act,
            nn.Dropout(p),
            nn.Linear(hidden_channel_dim, dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        y = self.ln1(x)
        y = self.token_linear(y.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop1(y)
        y = self.channel_mlp(self.ln2(x))
        return x + y

class ResMLP(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=12, hidden_channel_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        P = self.patch.num_patches
        self.blocks = nn.Sequential(*[
            ResMLPBlock(P, embed_dim, hidden_channel_dim, p=p, act=act) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.ln(x).mean(dim=1)
        return self.head(x)

class ASMLPBlock(nn.Module):
    def __init__(self, grid_size: int, dim: int, hidden_channel_dim: int, p=0.0, act='gelu'):
        super().__init__()
        self.grid = grid_size
        self.dim = dim
        assert dim % 4 == 0, "embed_dim must be divisible by 4"
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p)
        self.proj = nn.Linear(dim * 2, dim)
        the_act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_channel_dim),
            the_act,
            nn.Dropout(p),
            nn.Linear(hidden_channel_dim, dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        N, P, D = x.shape
        H = W = self.grid
        y = self.ln1(x).view(N, H, W, D)
        Cg = D // 4
        y1, y2, y3, y4 = y[..., :Cg], y[..., Cg:2*Cg], y[..., 2*Cg:3*Cg], y[..., 3*Cg:]
        y1 = torch.roll(y1, shifts=1, dims=1)
        y2 = torch.roll(y2, shifts=-1, dims=1)
        y3 = torch.roll(y3, shifts=1, dims=2)
        y4 = torch.roll(y4, shifts=-1, dims=2)
        y_shift = torch.cat([y1, y2, y3, y4], dim=-1).view(N, P, D)
        y = torch.cat([y_shift, x], dim=-1)
        y = self.proj(y)
        x = x + self.drop(y)
        y = self.channel_mlp(self.ln2(x))
        return x + y

class ASMLP(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, num_classes,
                 embed_dim=256, depth=12, hidden_channel_dim=1024, p=0.0, act='gelu'):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        grid = int(math.sqrt(self.patch.num_patches))
        self.blocks = nn.Sequential(*[
            ASMLPBlock(grid, embed_dim, hidden_channel_dim, p=p, act=act) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.ln(x).mean(dim=1)
        return self.head(x)


def build_image_mlp(model_name: str,
                    image_size: int, in_chans: int, num_classes: int,
                    patch_size: int,
                    embed_dim: int,
                    depth: int,
                    token_mlp_dim: int,
                    channel_mlp_dim: int) -> nn.Module:
    name = model_name.lower()
    p = FIXED["dropout_p"]
    act = FIXED["activation"]
    if name == 'mixer':
        return MLPMixer(image_size=image_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                        embed_dim=embed_dim, depth=depth, token_mlp_dim=token_mlp_dim,
                        channel_mlp_dim=channel_mlp_dim, p=p, act=act)
    elif name == 'resmlp':
        return ResMLP(image_size=image_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                      embed_dim=embed_dim, depth=depth, hidden_channel_dim=channel_mlp_dim, p=p, act=act)
    elif name == 'asmlp':
        if embed_dim % 4 != 0:
            raise ValueError("For AS-MLP, embed_dim must be divisible by 4.")
        return ASMLP(image_size=image_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                     embed_dim=embed_dim, depth=depth, hidden_channel_dim=channel_mlp_dim, p=p, act=act)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _safe_path_component(s: str, maxlen: int = 140) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s[:maxlen]

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gpu_info():
    if DEVICE.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        return prop.name
    return 'CPU'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

class SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.base[self.indices[i]]

# Reusable DataLoader builder
def build_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    if platform.system() == "Windows":
        num_workers = 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=(DEVICE.type=='cuda'), persistent_workers=False
    )


@torch.no_grad()
def _initial_loss(model: nn.Module, criterion: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.eval()
    logits = model(xb)
    return float(criterion(logits, yb).item())

def extract_probe_features(model: nn.Module, criterion: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, float]:
    model.to(DEVICE)
    model.train(False)
    xb = xb.to(DEVICE, non_blocking=True)
    yb = yb.to(DEVICE, non_blocking=True).long()

    model.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=False, retain_graph=False, allow_unused=True)
    flat_grads = [g.reshape(-1) for g in grads if g is not None]
    if len(flat_grads) == 0:
        gcat = torch.zeros(1, device=DEVICE)
    else:
        gcat = torch.cat(flat_grads)
    grad_norm = torch.norm(gcat, p=2).item()
    ntk_trace_proxy = torch.sum(gcat * gcat).item()
    init_loss = loss.item()
    return {
        'gradient_norm_log10': float(np.log10(grad_norm + 1e-8)),
        'ntk_trace_proxy_log10': float(np.log10(ntk_trace_proxy + 1e-8)),
        'initial_loss_log10': float(np.log10(init_loss + 1e-8)),
    }


def _build_optimizer(params, name: str, lr: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(
            params, lr=lr,
            betas=FIXED["adamw"]["betas"],
            eps=FIXED["adamw"]["eps"],
            weight_decay=FIXED["adamw"]["weight_decay"]
        )
    elif name == 'sgd':
        return torch.optim.SGD(
            params, lr=lr,
            momentum=FIXED["sgd"]["momentum"],
            weight_decay=FIXED["sgd"]["weight_decay"],
            nesterov=FIXED["sgd"]["nesterov"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def train_until_plateau(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        lr: float,
                        optimizer_name: str,
                        precision_bits: int,
                        max_epochs: int,
                        patience: int,
                        min_delta: float,
                        record_first_k: int = 20) -> Tuple[int, List[float]]:

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = _build_optimizer(model.parameters(), optimizer_name, lr)
    use_amp = (precision_bits == 16 and DEVICE.type == 'cuda')
    scaler = _GradScaler(enabled=use_amp)

    best_val = float('inf')
    no_improve = 0
    val_curve: List[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train(True)
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).long()

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with _amp_autocast(True):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()


        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).long()
                logits = model(xb)
                vloss = criterion(logits, yb).item()
                val_loss += vloss
                n += 1
        val_loss /= max(1, n)
        if len(val_curve) < record_first_k:
            val_curve.append(float(val_loss))

        if val_loss < best_val - min_delta:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            return epoch, val_curve

    return max_epochs, val_curve


def run_single_mlp_experiment(
    dataset="CIFAR100",
    model="asmlp",
    data_root="./data",
    batch_size=128,
    lr=1e-3,
    optimizer="AdamW",
    precision=32,
    embed_dim=128,
    depth=12,
    patch_size=None,
    val_fraction=0.2,
    probe_batch_size=64,
    num_workers=0,
    seed=42,
):
    set_seed(seed)

    ds, num_classes, input_shape, ds_label, total_N = DATASET_BUILDERS[dataset](data_root)
    in_chans, H, W = input_shape


    if dataset == "CIFAR100":
        image_size = 32
        patch_size = patch_size or (4 if model == 'asmlp' else 8)
    elif dataset == "TinyImageNet":
        image_size = 64
        patch_size = patch_size or 8
    elif dataset == "STL10":
        image_size = 96
        patch_size = patch_size or 8
    else:
        raise ValueError("Unsupported dataset")


    if dataset == "TinyImageNet":
        train_ds = ds
        val_ds = TinyImageNet200(root=data_root, split="val", transform=train_ds.transform)
    else:
        n_total = len(ds)
        n_val = int(round(val_fraction * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


    probe_loader = build_loader(train_ds, batch_size=probe_batch_size, shuffle=True, num_workers=num_workers)
    probe_xb, probe_yb = next(iter(probe_loader))
    probe_xb, probe_yb = probe_xb.to(DEVICE), probe_yb.to(DEVICE).long()


    channel_mlp_dim = 4 * embed_dim
    token_mlp_dim = max(128, embed_dim // 2)
    model_obj = build_image_mlp(
        model_name=model,
        image_size=image_size, in_chans=in_chans, num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, depth=depth,
        token_mlp_dim=token_mlp_dim,
        channel_mlp_dim=channel_mlp_dim
    )

    P_params = param_count(model_obj)
    criterion = nn.CrossEntropyLoss()
    probe_feats = extract_probe_features(model_obj, criterion, probe_xb, probe_yb)

    features = {
        'param_count_log10': float(np.log10(P_params + 1e-8)),
        'learning_rate_log10': float(np.log10(lr + 1e-12)),
        'batch_size_log10': float(np.log10(batch_size + 1e-12)),
        **probe_feats,
    }

    return features
