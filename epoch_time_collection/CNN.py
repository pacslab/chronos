import os, re, zipfile, urllib.request, time, platform, argparse, random, warnings
from typing import Callable, Tuple, Dict, Any, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root
    from torch.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = True

def _amp_autocast(enabled: bool):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    if _USE_TORCH_AMP_ROOT:
        return _autocast_root(device_type='cuda', dtype=torch.float16)
    else:
        return _autocast_cuda(dtype=torch.float16)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def gpu_name():
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    return 'CPU'

def gpu_props():
    if DEVICE.type != 'cuda':
        return dict(sm_count=0, total_vram_bytes=0)
    p = torch.cuda.get_device_properties(0)
    return dict(sm_count=getattr(p, 'multi_processor_count', 0),
                total_vram_bytes=getattr(p, 'total_memory', 0))

def _safe(s: str, maxlen:int=140):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s[:maxlen]


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


def build_loader(ds, batch_size:int, num_workers:int=2) -> DataLoader:
    if platform.system() == "Windows":
        num_workers = 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
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


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mem_bytes_for_params(param_cnt:int) -> int:
    return param_cnt * 4

def optimizer_state_multiplier(name:str) -> int:
    n = name.lower()
    if n in ['adam','adamw']: return 2
    if n in ['rmsprop','adagrad','sgd']: return 1
    return 1

def count_depth(model: nn.Module) -> int:
    depth = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            depth += 1
    return depth

class StatEstimator:
    def __init__(self, count_acts_for=(nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
        self.handles = []
        self.flops_total: float = 0.0
        self.flops_1x1: float = 0.0
        self.flops_3x3: float = 0.0
        self.flops_depthwise: float = 0.0
        self.flops_grouped: float = 0.0
        self.act_scalars: int = 0
        self.sum_hwC: int = 0
        self.max_hwC: int = 0
        self.count_acts_for = count_acts_for
        self.bn_layers = 0
        self.act_layers = 0

    def _hook(self, module, inputs, output):
        try:
            if isinstance(module, nn.Conv2d):
                x = inputs[0]
                if x.ndim == 4 and isinstance(output, torch.Tensor) and output.ndim == 4:
                    N, Cin, Hin, Win = x.shape
                    Cout, Hout, Wout = output.shape[1], output.shape[2], output.shape[3]
                    Kh, Kw = module.kernel_size
                    groups = max(1, getattr(module, "groups", 1))
                    flops = 2.0 * (Cin / groups) * Cout * Kh * Kw * Hout * Wout
                    self.flops_total += flops

                    if groups == Cin and Cout == Cin:  # depthwise
                        self.flops_depthwise += flops
                        self.flops_grouped += flops
                    elif groups > 1:
                        self.flops_grouped += flops
                        if Kh == 3 and Kw == 3:
                            self.flops_3x3 += flops
                    else:
                        if Kh == 1 and Kw == 1:
                            self.flops_1x1 += flops
                        elif Kh == 3 and Kw == 3:
                            self.flops_3x3 += flops


                    hwc = int(Hout * Wout * Cout)
                    self.sum_hwC += hwc
                    if hwc > self.max_hwC:
                        self.max_hwC = hwc

            elif isinstance(module, nn.Linear):
                self.flops_total += 2.0 * module.in_features * module.out_features

            if isinstance(module, nn.BatchNorm2d):
                self.bn_layers += 1
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish, nn.GELU)):
                self.act_layers += 1

        except Exception:
            pass

        try:
            if isinstance(module, self.count_acts_for) and isinstance(output, torch.Tensor):
                per_sample = int(np.prod(output.shape[1:]))  # exclude batch dim
                self.act_scalars += per_sample
        except Exception:
            pass

    def attach(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                              nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish, nn.GELU,
                              nn.MaxPool2d, nn.AvgPool2d)):
                self.handles.append(m.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()

    def reset(self):
        self.flops_total = 0.0
        self.flops_1x1 = 0.0
        self.flops_3x3 = 0.0
        self.flops_depthwise = 0.0
        self.flops_grouped = 0.0
        self.act_scalars = 0
        self.sum_hwC = 0
        self.max_hwC = 0
        self.bn_layers = 0
        self.act_layers = 0

    def as_features(self) -> Dict[str, Any]:
        tot = self.flops_total if self.flops_total > 0 else 1.0
        return dict(
            flops_fwd_per_sample=float(self.flops_total),
            flops_1x1_frac=float(self.flops_1x1 / tot),
            flops_3x3_frac=float(self.flops_3x3 / tot),
            flops_depthwise_frac=float(self.flops_depthwise / tot),
            flops_grouped_frac=float(self.flops_grouped / tot),
            act_scalars=int(self.act_scalars),
            sum_hwC=int(self.sum_hwC),
            max_hwC=int(self.max_hwC),
            bn_layers=int(self.bn_layers),
            act_layers=int(self.act_layers),
        )


def _build_optimizer(params, name: str, lr: float, weight_decay: float):
    n = name.lower()
    if n == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif n == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def measure_batch_timing(model: nn.Module,
                         train_loader: DataLoader,
                         warmup_batches: int,
                         measure_batches: int,
                         lr: float,
                         optimizer_name: str,
                         weight_decay: float,
                         amp: bool,
                         channels_last: bool,
                         compile_model: bool) -> Dict[str, Any]:
    model = model.to(DEVICE)
    if channels_last and DEVICE.type == 'cuda':
        model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    opt = _build_optimizer(model.parameters(), optimizer_name, lr, weight_decay)

    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    scaler = _GradScaler(enabled=(amp and DEVICE.type=='cuda'))

    total_batches = warmup_batches + measure_batches
    it = iter(train_loader)
    model.train()

    total_step_ms: List[float] = []
    loader_ms_list: List[float] = []
    compute_ms_list: List[float] = []

    for i in range(total_batches):

        t0_load = time.perf_counter()
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader); xb, yb = next(it)
        t1_load = time.perf_counter()
        loader_ms = (t1_load - t0_load) * 1000.0

        xb = xb.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == 'cuda':
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(DEVICE, non_blocking=True).long()

        opt.zero_grad(set_to_none=True)

        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t0_compute = time.perf_counter()

        if scaler.is_enabled():
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

        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t1_compute = time.perf_counter()

        compute_ms = (t1_compute - t0_compute) * 1000.0
        total_ms = loader_ms + compute_ms

        if i >= warmup_batches:
            total_step_ms.append(total_ms)
            loader_ms_list.append(loader_ms)
            compute_ms_list.append(compute_ms)


    avg_ms = float(np.mean(total_step_ms)) if total_step_ms else float('nan')
    p90_ms = float(np.percentile(total_step_ms, 90)) if total_step_ms else float('nan')
    p50_ms = float(np.percentile(total_step_ms, 50)) if total_step_ms else float('nan')
    std_ms = float(np.std(total_step_ms, ddof=0)) if total_step_ms else float('nan')

    avg_loader_ms = float(np.mean(loader_ms_list)) if loader_ms_list else float('nan')
    avg_compute_ms = float(np.mean(compute_ms_list)) if compute_ms_list else float('nan')
    loader_ratio = float(avg_loader_ms / max(1e-9, (avg_loader_ms + avg_compute_ms))) if np.isfinite(avg_loader_ms) else float('nan')

    return {
        'avg_batch_ms': avg_ms,
        'p90_batch_ms': p90_ms,
        'p50_batch_ms': p50_ms,
        'std_batch_ms': std_ms,
        'avg_loader_ms': avg_loader_ms,
        'avg_compute_ms': avg_compute_ms,
        'loader_ratio': loader_ratio,
    }


def main():
    ap = argparse.ArgumentParser("CNN Benchmark (Per-Batch Timing) + Extended Features — mapping enforced")
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--logdir', type=str, default='./logs')


    ap.add_argument('--models', nargs='+', default=['resnet50','vgg16','mobilenetv2','densenet121'])


    ap.add_argument('--datasets', nargs='+', default=['CIFAR10','CIFAR100','TinyImageNet','STL10'])


    ap.add_argument('--batch-sizes', nargs='+', type=int, default=[32,64,128,256])
    ap.add_argument('--lrs', nargs='+', type=float, default=[5e-4, 1e-3, 2e-3])
    ap.add_argument('--optimizers', nargs='+', default=['SGD','AdamW'])
    ap.add_argument('--precisions', nargs='+', type=int, default=[16,32], choices=[16,32])


    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--warmup-batches', type=int, default=5)
    ap.add_argument('--measure-batches', type=int, default=20)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)


    ap.add_argument('--compile', action='store_true', help='Attempt torch.compile(models)')
    ap.add_argument('--channels-last', action='store_true', help='Use channels_last memory format')
    ap.add_argument('--cudnn-benchmark', action='store_true', help='Enable torch.backends.cudnn.benchmark')

    args = ap.parse_args()
    set_seed(args.seed)
    if args.cudnn_benchmark and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


    ALLOWED_BY_MODEL = {
        'resnet50':    ['TinyImageNet', 'CIFAR10'],
        'vgg16':       ['TinyImageNet', 'CIFAR100'],
        'mobilenetv2': ['CIFAR10', 'STL10'],
        'densenet121': ['CIFAR100', 'TinyImageNet'],
    }

    os.makedirs(args.logdir, exist_ok=True)
    gname = gpu_name()
    props = gpu_props()

    out_csv = os.path.join(os.path.abspath(args.logdir),
                           f"CNN_{_safe(gname) if DEVICE.type=='cuda' else 'CPU'}.csv")

    rows = []


    built_ds: Dict[str, Any] = {}
    for ds_name in args.datasets:
        if ds_name not in DATASET_BUILDERS:
            print(f"[WARN] Unknown dataset: {ds_name} (skipping)")
            continue
        ds, num_classes, input_shape, ds_label, total_N = DATASET_BUILDERS[ds_name](args.data_root)
        built_ds[ds_label] = (ds, num_classes, input_shape, total_N)

    for model_name in args.models:
        allowed = [d for d in ALLOWED_BY_MODEL.get(model_name.lower(), []) if d in built_ds.keys()]
        if not allowed:
            print(f"[WARN] No allowed datasets available for model '{model_name}'. Skipping.")
            continue

        for ds_label in allowed:
            ds, num_classes, input_shape, total_N = built_ds[ds_label]
            in_ch, H, W = input_shape


            template_model = build_cnn(model_name, num_classes).to(DEVICE)
            template_model.eval()

            P_params = param_count(template_model)
            depth_layers = count_depth(template_model)

            stat = StatEstimator()
            stat.attach(template_model); stat.reset()
            sample_input = torch.zeros(1, in_ch, H, W, device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
                _ = template_model(sample_input)
                if DEVICE.type == 'cuda': torch.cuda.synchronize()
            feats = stat.as_features()
            stat.detach()
            del template_model

            flops_fwd_per_sample = feats['flops_fwd_per_sample']
            act_scalars = feats['act_scalars']
            param_bytes = mem_bytes_for_params(P_params)


            loader_cache: Dict[int, DataLoader] = {}


            grid = []
            for opt_name in args.optimizers:
                for lr in args.lrs:
                    for B in args.batch_sizes:
                        for precision in args.precisions:
                            grid.append((opt_name, lr, B, precision))

            for opt_name, lr, B, precision in tqdm(grid, desc=f"{ds_label}-{model_name}", unit="trial"):
                if B not in loader_cache:
                    try:
                        loader_cache[B] = build_loader(ds, batch_size=B, num_workers=args.num_workers)
                    except RuntimeError as e:
                        print(f"[ERROR] DataLoader failed for B={B}: {e}")
                        continue
                loader = loader_cache[B]


                model = build_cnn(model_name, num_classes)

                use_amp = (precision == 16 and DEVICE.type=='cuda')
                opt_mult = optimizer_state_multiplier(opt_name)
                optimizer_state_bytes = param_bytes * opt_mult
                master_param_bytes = param_bytes if use_amp else 0
                total_state_bytes = param_bytes + optimizer_state_bytes + master_param_bytes


                act_bytes_per_scalar = 2 if precision == 16 else 4
                act_bytes_per_sample_proxy = int(act_scalars * act_bytes_per_scalar)


                train_flops_multiplier = 3.0
                flops_train_per_sample = flops_fwd_per_sample * train_flops_multiplier
                flops_train_per_batch = flops_train_per_sample * B
                ai_train = float(flops_train_per_sample / max(1.0, (param_bytes + act_bytes_per_sample_proxy)))

                try:
                    out = measure_batch_timing(
                        model=model,
                        train_loader=loader,
                        warmup_batches=args.warmup_batches,
                        measure_batches=args.measure_batches,
                        lr=lr,
                        optimizer_name=opt_name,
                        weight_decay=args.weight_decay,
                        amp=use_amp,
                        channels_last=args.channels_last,
                        compile_model=args.compile
                    )
                except RuntimeError as e:
                    print(f"[ERROR] Training failed [{model_name}] on {ds_label} (B={B}, lr={lr}, opt={opt_name}): {e}")
                    del model
                    if DEVICE.type=='cuda': torch.cuda.empty_cache()
                    continue


                compute_capacity_proxy = int(props['sm_count']) * (2 if precision == 16 else 1)

                row = {

                    'dataset': ds_label,
                    'input_shape': str(input_shape),
                    'num_classes': int(num_classes),
                    'dataset_size': int(total_N),


                    'optimizer': opt_name,
                    'learning_rate': float(lr),
                    'batch_size': int(B),
                    'precision': int(precision),
                    'amp': bool(use_amp),


                    'model': model_name,
                    'architecture': 'CNN',
                    'depth': int(depth_layers),


                    'param_count': int(P_params),
                    'param_bytes': int(param_bytes),
                    'optimizer_state_bytes': int(optimizer_state_bytes),
                    'master_param_bytes': int(master_param_bytes),
                    'total_state_bytes': int(total_state_bytes),


                    'flops_fwd_per_sample': float(flops_fwd_per_sample),
                    'flops_train_per_sample': float(flops_train_per_sample),
                    'flops_train_per_batch': float(flops_train_per_batch),
                    'act_bytes_per_sample_proxy': int(act_bytes_per_sample_proxy),
                    'arithmetic_intensity_train': float(ai_train),


                    'flops_1x1_frac': float(feats['flops_1x1_frac']),
                    'flops_3x3_frac': float(feats['flops_3x3_frac']),
                    'flops_depthwise_frac': float(feats['flops_depthwise_frac']),
                    'flops_grouped_frac': float(feats['flops_grouped_frac']),
                    'bn_layers': int(feats['bn_layers']),
                    'act_layers': int(feats['act_layers']),
                    'sum_hwC': int(feats['sum_hwC']),
                    'max_hwC': int(feats['max_hwC']),


                    'avg_batch_time_ms': float(out['avg_batch_ms']),
                    'p90_batch_time_ms': float(out['p90_batch_ms']),

                    'loader_wait_ms': float(out['avg_loader_ms']),
                    'compute_ms': float(out['avg_compute_ms']),
                    'loader_ratio': float(out['loader_ratio']),


                    'gpu_name': gname if DEVICE.type=='cuda' else 'CPU'
                }
                rows.append(row)

                del model
                if DEVICE.type=='cuda':
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved benchmark to:\n  {out_csv}")

if __name__ == "__main__":
    main()
