import os, re, math, time, json, random, argparse, platform, warnings, zipfile, urllib.request
from typing import Tuple, Callable, Dict, Any, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def gpu_info():
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    return 'CPU'


try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root
    from torch.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = True

from contextlib import nullcontext
def _amp_autocast(enabled: bool):
    if not enabled: return nullcontext()
    if _USE_TORCH_AMP_ROOT:
        return _autocast_root(device_type='cuda', dtype=torch.float16)
    else:
        return _autocast_cuda(dtype=torch.float16)


_TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
_TINY_ZIP = "tiny-imagenet-200.zip"
_TINY_DIR = "tiny-imagenet-200"

def _download_tiny(root: str) -> str:
    root = os.path.abspath(root); os.makedirs(root, exist_ok=True)
    target = os.path.join(root, _TINY_DIR)
    if os.path.isdir(target) and os.path.isdir(os.path.join(target, "train")):
        return target
    zip_path = os.path.join(root, _TINY_ZIP)
    if not os.path.exists(zip_path):
        print("Downloading TinyImageNet-200 (~236MB)...")
        urllib.request.urlretrieve(_TINY_URL, zip_path)
    print("Extracting TinyImageNet-200...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return target

class TinyImageNet200(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Callable = None):
        assert split in ("train", "val")
        self.root = _download_tiny(root)
        self.split = split
        self.transform = transform
        with open(os.path.join(self.root, "wnids.txt"), "r") as f:
            self.wnids = [x.strip() for x in f if x.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            tdir = os.path.join(self.root, "train")
            for wnid in self.wnids:
                imgd = os.path.join(tdir, wnid, "images")
                if not os.path.isdir(imgd): continue
                for fn in os.listdir(imgd):
                    if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(imgd, fn), self.class_to_idx[wnid]))
        else:
            vdir = os.path.join(self.root, "val")
            ann = os.path.join(vdir, "val_annotations.txt")
            mapping = {}
            with open(ann, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
            imgd = os.path.join(vdir, "images")
            for fn in os.listdir(imgd):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    wnid = mapping.get(fn, None)
                    if wnid is None: continue
                    self.samples.append((os.path.join(imgd, fn), self.class_to_idx[wnid]))
        self.num_classes = 200
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, target


def build_cifar10(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    return ds, 10, (3,32,32), 'CIFAR10', len(ds)

def build_cifar100(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    ds = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
    return ds, 100, (3,32,32), 'CIFAR100', len(ds)

def build_tiny_imagenet(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821))
    ])
    ds = TinyImageNet200(root=root, split="train", transform=tfm)
    return ds, 200, (3,64,64), 'TinyImageNet', len(ds)

def build_stl10(root):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713))
    ])
    ds = datasets.STL10(root=root, split='train', download=True, transform=tfm)
    return ds, 10, (3,96,96), 'STL10', len(ds)

DATASET_BUILDERS = {
    'CIFAR10': build_cifar10,
    'CIFAR100': build_cifar100,
    'TinyImageNet': build_tiny_imagenet,
    'STL10': build_stl10,
}


def build_loader(ds, batch_size:int, shuffle:bool, num_workers:int=2) -> DataLoader:
    if platform.system() == "Windows":
        num_workers = 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=(DEVICE.type=='cuda'), persistent_workers=False
    )


def make_vgg16(num_classes:int) -> nn.Module:
    m = models.vgg16_bn(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def make_resnet50(num_classes:int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_densenet121(num_classes:int) -> nn.Module:
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def make_mobilenetv2(num_classes:int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def build_cnn(model_name:str, num_classes:int) -> nn.Module:
    n = model_name.lower()
    if n == 'vgg16':        return make_vgg16(num_classes)
    if n == 'resnet50':     return make_resnet50(num_classes)
    if n == 'densenet121':  return make_densenet121(num_classes)
    if n == 'mobilenetv2':  return make_mobilenetv2(num_classes)
    raise ValueError(f"Unknown CNN model: {model_name}")


def _safe_path_component(s: str, maxlen: int = 140) -> str:
    s = re.sub(r'[<>:\"/\\|?*\x00-\x1F]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s[:maxlen]

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                create_graph=False, retain_graph=False, allow_unused=True)
    flat_grads = [g.reshape(-1) for g in grads if g is not None]
    gcat = torch.cat(flat_grads) if len(flat_grads) else torch.zeros(1, device=DEVICE)
    grad_norm = torch.norm(gcat, p=2).item()
    ntk_trace_proxy = torch.sum(gcat * gcat).item()
    init_loss = loss.item()
    return {
        'gradient_norm_log10': float(np.log10(grad_norm + 1e-8)),
        'ntk_trace_proxy_log10': float(np.log10(ntk_trace_proxy + 1e-8)),
        'initial_loss_log10': float(np.log10(init_loss + 1e-8)),
    }

# ------------------- Optimizer builder -------------------
def _build_optimizer(params, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def run_single_cnn_experiment(
    dataset="CIFAR10",
    model="resnet50",
    data_root="./data",
    batch_size=128,
    lr=1e-3,
    optimizer="AdamW",
    precision=32,
    val_fraction=0.2,
    probe_batch_size=64,
    num_workers=0,
    seed=42,
):
    set_seed(seed)


    ds, num_classes, input_shape, ds_label, total_N = DATASET_BUILDERS[dataset](data_root)
    in_ch, H, W = input_shape


    if dataset == "TinyImageNet":
        train_ds = ds
        val_ds = TinyImageNet200(root=data_root, split="val", transform=train_ds.transform)
    else:
        n_total = len(ds)
        n_val = int(round(val_fraction * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(seed))


    probe_loader = build_loader(train_ds, batch_size=probe_batch_size, shuffle=True, num_workers=num_workers)
    probe_xb, probe_yb = next(iter(probe_loader))
    probe_xb, probe_yb = probe_xb.to(DEVICE), probe_yb.to(DEVICE).long()


    model_obj = build_cnn(model, num_classes)
    P_params = param_count(model_obj)
    criterion = nn.CrossEntropyLoss()


    probe_feats = extract_probe_features(model_obj, criterion, probe_xb, probe_yb)


    features = {
        "param_count_log10": float(np.log10(P_params + 1e-8)),
        "learning_rate_log10": float(np.log10(lr + 1e-12)),
        "batch_size_log10": float(np.log10(batch_size + 1e-12)),
        **probe_feats,
    }

    return features
