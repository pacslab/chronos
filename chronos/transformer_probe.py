import os, re, math, json, time, random, warnings, argparse, platform
from typing import List, Tuple, Dict, Any, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_TQDM = dict(dynamic_ncols=True, mininterval=0.2, smoothing=0.1, leave=True)
def tbar(it=None, **kw):
    p = _TQDM.copy(); p.update(kw); from tqdm.auto import tqdm as _t; return _t(it, **p) if it is not None else _t(**p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_ROOT_AMP = False
except Exception:
    from torch.amp import autocast as _autocast_root
    from torch.amp import GradScaler as _GradScaler
    _USE_ROOT_AMP = True

def amp_cast(enabled: bool):
    from contextlib import nullcontext
    if not enabled: return nullcontext()
    return _autocast_root(device_type='cuda', dtype=torch.float16) if _USE_ROOT_AMP else _autocast_cuda(dtype=torch.float16)

def set_device(gpu_id: Optional[int]):
    global DEVICE
    if torch.cuda.is_available() and gpu_id is not None and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        DEVICE = torch.device(f"cuda:{gpu_id}")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE

def parse_gpu_map(s: Optional[str]) -> Dict[str, int]:

    if not s:
        return {}
    out: Dict[str,int] = {}
    for pair in s.split(","):
        k, v = pair.split(":")
        out[k.strip()] = int(v.strip())
    return out

DEFAULT_SEED = 42
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


try:
    import datasets as hfds
    _HAS_HF = True
except Exception:
    _HAS_HF = False

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

try:
    import torchvision
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

BOS, EOS, PAD, UNK = "<bos>", "<eos>", "<pad>", "<unk>"
_word_re = re.compile(r"\w+([’']\w+)?", re.UNICODE)
def basic_split(text: str) -> List[str]:
    return [m.group(0).lower() for m in _word_re.finditer(text or "")]

class Vocab:
    def __init__(self, stoi:Dict[str,int], unk_token:str=UNK):
        self.stoi = dict(stoi)
        self.itos = [None] * len(stoi)
        for k,v in self.stoi.items(): self.itos[v] = k
        self.unk = unk_token; self.pad = PAD
        self._unk_idx = self.stoi.get(unk_token, 0)
    def __len__(self): return len(self.stoi)
    def __getitem__(self, s:str) -> int: return self.stoi.get(s, self._unk_idx)

def build_vocab(text_iter, max_tokens:int=30000, specials:List[str]=[UNK, PAD, BOS, EOS]) -> Vocab:
    freq: Dict[str,int] = {}
    for txt in text_iter:
        for tok in basic_split(txt): freq[tok] = freq.get(tok, 0) + 1
    stoi = {}
    for sp in specials:
        if sp not in stoi: stoi[sp] = len(stoi)
    for tok, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
        if tok in stoi: continue
        if len(stoi) >= max_tokens: break
        stoi[tok] = len(stoi)
    return Vocab(stoi)

def _ensure_nonempty(ids: List[int], vocab: Vocab) -> List[int]:
    return ids if ids else [vocab[UNK]]

def _need_hf():
    if not _HAS_HF: raise RuntimeError("Hugging Face 'datasets' not installed/available.")

def build_imdb(root, max_seq_len:int, vocab_size:int):
    _need_hf()
    ds = hfds.load_dataset("imdb", split="train", cache_dir=root)
    N = len(ds)
    vocab = build_vocab((ds[i]["text"] for i in range(N)), max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        ids = [vocab[t] for t in basic_split(ds[i]["text"])]
        pairs.append(( _ensure_nonempty(ids, vocab)[:max_seq_len], int(ds[i]["label"]) ))
    num_classes=2; label="IMDB"
    return pairs, vocab, num_classes, N, label

def build_sst2(root, max_seq_len:int, vocab_size:int):
    _need_hf()
    ds = hfds.load_dataset("glue", "sst2", split="train", cache_dir=root)
    N = len(ds)
    vocab = build_vocab((ds[i]["sentence"] for i in range(N)), max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        ids = [vocab[t] for t in basic_split(ds[i]["sentence"])]
        pairs.append(( _ensure_nonempty(ids, vocab)[:max_seq_len], int(ds[i]["label"]) ))
    num_classes=2; label="SST2"
    return pairs, vocab, num_classes, N, label

def to_text_cls_dataset(pairs, vocab, max_seq_len:int, num_classes:int):
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, i): return pairs[i]
        @staticmethod
        def collate(batch, pad_idx=vocab[PAD]):
            seqs, labels = zip(*batch)
            L = [min(len(s), max_seq_len) for s in seqs]
            L = torch.tensor(L, dtype=torch.long)
            X = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            for i, s in enumerate(seqs):
                s = s[:max_seq_len] if s else [pad_idx]
                X[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.long)
            return X, L, y
    d = _DS(); d.collate_fn = _DS.collate; return d

def build_wmt14(root, max_seq_len:int, vocab_size:int):
    _need_hf()
    ds = hfds.load_dataset("wmt14", "de-en", split="train", cache_dir=root)
    N = len(ds)
    srcs = [ds[i]["translation"]["de"] for i in range(N)]
    tgts = [ds[i]["translation"]["en"] for i in range(N)]
    vocab = build_vocab(srcs + tgts, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        de = [vocab[t] for t in basic_split(srcs[i])]
        en = [vocab[t] for t in basic_split(tgts[i])]
        de = [vocab[BOS]] + _ensure_nonempty(de, vocab) + [vocab[EOS]]
        en = [vocab[BOS]] + _ensure_nonempty(en, vocab) + [vocab[EOS]]
        pairs.append((de[:max_seq_len], en[:max_seq_len]))
    return pairs, vocab, N, "WMT14_EN_DE"

def build_cnndm(root, max_seq_len:int, vocab_size:int):
    _need_hf()
    ds = hfds.load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=root)
    N = len(ds)
    srcs=[]; tgts=[]
    for i in range(N):
        ex = ds[i]
        srcs.append(ex.get("article") or ex.get("document") or "")
        tgts.append(ex.get("highlights") or ex.get("summary") or "")
    vocab = build_vocab(srcs + tgts, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        x = [vocab[t] for t in basic_split(srcs[i])]
        y = [vocab[t] for t in basic_split(tgts[i])]
        x = [vocab[BOS]] + _ensure_nonempty(x, vocab) + [vocab[EOS]]
        y = [vocab[BOS]] + _ensure_nonempty(y, vocab) + [vocab[EOS]]
        pairs.append((x[:max_seq_len], y[:max_seq_len]))
    return pairs, vocab, N, "CNNDM_SUM"

def to_s2s_dataset(pairs, vocab, max_seq_len:int):
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, i): return pairs[i]
        @staticmethod
        def collate(batch, pad_idx=vocab[PAD]):
            srcs, tgts = zip(*batch)
            Ls = [min(len(s), max_seq_len) for s in srcs]
            Lt = [min(len(t), max_seq_len) for t in tgts]
            Ls = torch.tensor(Ls, dtype=torch.long)
            Lt = torch.tensor(Lt, dtype=torch.long)
            Xs = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            Xt = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            for i,(s,t) in enumerate(zip(srcs, tgts)):
                s = s[:max_seq_len] if s else [pad_idx]
                t = t[:max_seq_len] if t else [pad_idx]
                Xs[i,:len(s)] = torch.tensor(s, dtype=torch.long)
                Xt[i,:len(t)] = torch.tensor(t, dtype=torch.long)
            return Xs, Ls, Xt, Lt
    d = _DS(); d.collate_fn = _DS.collate; return d

def build_vision(root, name:str, img_size:int=224):
    if not _HAS_TORCHVISION: raise RuntimeError("torchvision not available.")
    tfm = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    if name == "CIFAR100":
        ds = torchvision.datasets.CIFAR100(root=os.path.join(root,"cifar100"), train=True, download=True, transform=tfm); nc=100
    elif name == "STL10":
        ds = torchvision.datasets.STL10(root=os.path.join(root,"stl10"), split="train", download=True, transform=tfm); nc=10
    elif name == "TinyImageNet":
        if _HAS_HF:
            d = hfds.load_dataset("Maysee/tiny-imagenet", split="train", cache_dir=root)
            class Tiny(torch.utils.data.Dataset):
                def __len__(self): return len(d)
                def __getitem__(self, i):
                    ex=d[i]; img=ex["image"].convert("RGB")
                    return tfm(img), int(ex["label"])
            ds = Tiny(); nc=200
        else:
            raise RuntimeError("TinyImageNet requires HF datasets (Maysee/tiny-imagenet).")
    else:
        raise RuntimeError(f"Unknown vision dataset {name}")
    return ds, nc, len(ds)


class DistilTextClassifier(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=256, nhead:int=8, depth:int=6, d_ff:int=1024, num_classes:int=2, pad_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.pad_idx = pad_idx
    def forward(self, X, L):
        mask = (X != self.pad_idx)
        H = self.embedding(X)
        H = self.encoder(H, src_key_padding_mask=~mask)
        H = self.norm(H)
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask.sum(dim=1, keepdim=True), min=1).to(H.dtype)
        return self.fc(pooled)

class VanillaSeq2Seq(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, nhead:int=8, depth:int=6, d_ff:int=2048, pad_idx:int=0):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tf = nn.Transformer(d_model=d_model, nhead=nhead,
                                 num_encoder_layers=depth, num_decoder_layers=depth,
                                 dim_feedforward=d_ff, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx
    def forward(self, Xs, Ls, Xt, Lt):
        src_mask = (Xs == self.pad_idx); Tt = int(Lt.max().item())
        Xt = Xt[:, :Tt]
        tgt_mask = (Xt == self.pad_idx)
        causal = nn.Transformer.generate_square_subsequent_mask(Tt).to(Xs.device)
        hs = self.src_emb(Xs); ht = self.tgt_emb(Xt)
        out = self.tf(hs, ht, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask, tgt_mask=causal)
        return self.fc(out), Tt

def build_deit_tiny(num_classes:int):
    if not _HAS_TIMM: raise RuntimeError("timm is required for DeiT/ViT.")
    return timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

def build_vit_base(num_classes:int):
    if not _HAS_TIMM: raise RuntimeError("timm is required for DeiT/ViT.")
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)


class _Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, eps=1e-30, beta2=0.999, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, eps=eps, beta2=beta2, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, eps, beta2, wd = group["lr"], group["eps"], group["beta2"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if wd != 0: g = g.add(p, alpha=wd)
                st = self.state[p]
                if "v" not in st: st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                v = st["v"]; v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                upd = g / (v.sqrt() + eps)
                p.add_(upd, alpha=-lr)
        return loss

def build_optimizer(name:str, params, lr:float, weight_decay:float=0.0):
    n = name.lower()
    if n == "adamw": return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9,0.999), eps=1e-8)
    if n == "adam":  return torch.optim.Adam(params,  lr=lr, weight_decay=weight_decay, betas=(0.9,0.999), eps=1e-8)
    if n == "adafactor": return _Adafactor(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")


def build_loader(ds, batch_size:int, shuffle:bool=True, num_workers:int=2, collate_fn=None):
    if platform.system() == "Windows": num_workers = 0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(DEVICE.type=='cuda'), persistent_workers=False,
                      collate_fn=collate_fn)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def extract_probe_features(model: nn.Module, criterion: nn.Module, batch, task: str, pad_idx: Optional[int]) -> Dict[str, float]:
    model.to(DEVICE); model.train(False)
    with torch.enable_grad():
        if task == "text_cls":
            X, L, y = batch
            X = X.to(DEVICE, non_blocking=True); L = L.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            logits = model(X, L); loss = criterion(logits, y)
        elif task == "vision_cls":
            X, y = batch
            X = X.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            logits = model(X); loss = criterion(logits, y)
        else:  # seq2seq
            Xs, Ls, Xt, Lt = batch
            Xs = Xs.to(DEVICE, non_blocking=True); Xt = Xt.to(DEVICE, non_blocking=True)
            Ls = Ls.to(DEVICE, non_blocking=True); Lt = Lt.to(DEVICE, non_blocking=True)
            logits, Tt = model(Xs, Ls, Xt, Lt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), Xt[:, :Tt].reshape(-1))

        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                    create_graph=False, retain_graph=False, allow_unused=True)
        gflat = [g.reshape(-1) for g in grads if g is not None]
        gcat = torch.cat(gflat) if gflat else torch.zeros(1, device=DEVICE)
        grad_norm = torch.norm(gcat, p=2).item()
        ntk_trace_proxy = torch.sum(gcat * gcat).item()
        init_loss = loss.item()
    return {
        "gradient_norm_log10": float(np.log10(grad_norm + 1e-8)),
        "ntk_trace_proxy_log10": float(np.log10(ntk_trace_proxy + 1e-8)),
        "initial_loss_log10": float(np.log10(init_loss + 1e-8)),
    }


DATASET_SPECS = {
    "CIFAR100": {"num_classes": 100, "dataset_size": 50000},
    "CIFAR10": {"num_classes": 10, "dataset_size": 50000},
    "TinyImageNet": {"num_classes": 200, "dataset_size": 100000},
    "STL10": {"num_classes": 10, "dataset_size": 5000},
    "SST2": {"dataset_size": 67349, "seq_len": 128, "src_len": 128, "tgt_len": float('nan')},
    "IMDB": {"dataset_size": 25000, "seq_len": 128, "src_len": 128, "tgt_len": float('nan')},
    "WMT14_EN_DE": {"dataset_size": 4508785, "seq_len": float('nan'), "src_len": 128, "tgt_len": 128},
    "CNNDM_SUM": {"dataset_size": 287113, "seq_len": float('nan'), "src_len": 512, "tgt_len": 512},
}

def infer_vocab_size(dataset: str, model: str) -> int | None:
    text_datasets = {"SST2", "IMDB"}
    seq2seq_datasets = {"WMT14_EN_DE", "CNNDM_SUM"}
    vision_datasets = {"CIFAR10", "CIFAR100", "TinyImageNet", "STL10"}

    if dataset in text_datasets and model == "DistilBERT":
        return 30000
    elif dataset in seq2seq_datasets and model == "Transformer":
        return 30000
    elif dataset in vision_datasets and model in {"DeiT-Tiny", "ViT"}:
        return None
    else:

        return 30000 if "bert" in model.lower() or "transformer" in model.lower() else None


def run_single_transformer_experiment(
    model="DistilBERT",
    dataset="SST2",
    data_root="./data",
    batch_size=16,
    lr=1e-4,
    optimizer="AdamW",
    num_workers=0,
    seed=42,
    min_delta=1e-4,
    max_epochs_text=50,
    max_epochs_vision=120,
    max_epochs_s2s=80,
    precision=16,
):

    set_seed(seed)
    vocab_size = infer_vocab_size(dataset, model)


    if dataset in ("SST2", "IMDB"):
        pairs, vocab, nc, N, label = (
            build_sst2(data_root, 128, vocab_size)
            if dataset == "SST2"
            else build_imdb(data_root, 128, vocab_size)
        )
        ds = to_text_cls_dataset(pairs, vocab, 128, nc)
        task = "text_cls"; pad_idx = vocab[PAD]; collate = ds.collate_fn
        num_classes = nc
        max_epochs_used = max_epochs_text

    elif dataset in ("WMT14_EN_DE", "CNNDM_SUM"):
        if dataset == "WMT14_EN_DE":
            pairs, vocab, N, label = build_wmt14(data_root, 128, vocab_size)
        else:
            pairs, vocab, N, label = build_cnndm(data_root, 512, vocab_size)
        ds = to_s2s_dataset(pairs, vocab, 512)
        task = "seq2seq"; pad_idx = vocab[PAD]; collate = ds.collate_fn
        num_classes = None
        max_epochs_used = max_epochs_s2s

    elif dataset in ("CIFAR100", "STL10", "TinyImageNet", "CIFAR10"):
        ds, nc, N = build_vision(data_root, dataset, img_size=224)
        vocab = None; label = dataset; task = "vision_cls"; pad_idx = None; collate = None
        num_classes = nc
        max_epochs_used = max_epochs_vision

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


    probe_loader = build_loader(ds, batch_size=min(batch_size, 64),
                                shuffle=True, num_workers=num_workers, collate_fn=collate)
    probe_batch = next(iter(probe_loader))


    if task == "text_cls":
        model_obj = DistilTextClassifier(vocab_size=len(vocab), num_classes=num_classes, pad_idx=pad_idx)
    elif task == "vision_cls":
        if model == "DeiT-Tiny":
            model_obj = build_deit_tiny(num_classes=num_classes)
        else:
            model_obj = build_vit_base(num_classes=num_classes)
    elif task == "seq2seq":
        model_obj = VanillaSeq2Seq(vocab_size=len(vocab), pad_idx=pad_idx)
    else:
        raise ValueError(f"Unsupported task: {task}")


    criterion = (
        nn.CrossEntropyLoss(ignore_index=(pad_idx if (task == "seq2seq") else -100))
        if task == "seq2seq" else nn.CrossEntropyLoss()
    )
    probe_feats = extract_probe_features(model_obj, criterion, probe_batch, task, pad_idx)


    ds_info = DATASET_SPECS.get(dataset, {})
    num_classes = ds_info.get("num_classes", np.nan)
    dataset_size = ds_info.get("dataset_size", np.nan)

    P_params = param_count(model_obj)
    features = {
        "task": task,
        "param_count_log10": float(np.log10(P_params + 1e-8)),
        "learning_rate_log10": float(np.log10(lr + 1e-12)),
        "batch_size_log10": float(np.log10(batch_size + 1e-12)),
        "min_delta": float(min_delta),
        "max_epochs_used": int(max_epochs_used),
        **probe_feats,
    }

    return features

