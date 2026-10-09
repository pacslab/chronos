import os, re, time, argparse, random, warnings, math, platform, sys
from typing import List, Tuple, Dict, Any, Optional
from itertools import product

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


_TQDM_KW = dict(
    dynamic_ncols=True,
    mininterval=0.2,
    smoothing=0.1,
    disable=False,
    leave=True,
)

def tbar(iterable=None, **kw):
    params = _TQDM_KW.copy()
    params.update(kw)
    return tqdm(iterable, **params) if iterable is not None else tqdm(**params)


try:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScaler
    _USE_TORCH_AMP_ROOT = False
except Exception:
    from torch.amp import autocast as _autocast_root  # type: ignore
    from torch.amp import GradScaler as _GradScaler   # type: ignore
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
    torch.backends.cudnn.benchmark = True

def gpu_name():
    if DEVICE.type == 'cuda':
        return torch.cuda.get_device_properties(0).name
    return 'CPU'

def _safe(s: str, maxlen:int=140):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', str(s))
    s = re.sub(r'\s+', '_', s.strip())
    return s[:maxlen]

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
        for k,v in self.stoi.items():
            self.itos[v] = k
        self.unk = unk_token
        self.pad = PAD
        self._unk_idx = self.stoi.get(unk_token, 0)
    def __len__(self): return len(self.stoi)
    def __getitem__(self, item:str) -> int:
        return self.stoi.get(item, self._unk_idx)

def build_vocab_from_texts(text_iter, total:int, max_tokens:int=30000, specials:List[str]=[UNK, PAD, BOS, EOS]) -> Vocab:
    freq: Dict[str,int] = {}
    for txt in text_iter:
        for tok in basic_split(txt):
            freq[tok] = freq.get(tok, 0) + 1
    stoi = {}
    for sp in specials:
        if sp not in stoi:
            stoi[sp] = len(stoi)
    for tok, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
        if tok in stoi: continue
        if len(stoi) >= max_tokens: break
        stoi[tok] = len(stoi)
    return Vocab(stoi)

def _ensure_nonempty_ids(ids: List[int], vocab: Vocab) -> List[int]:
    return ids if len(ids) > 0 else [vocab[UNK]]

def _hf_require():
    if not _HAS_HF:
        raise RuntimeError("Hugging Face 'datasets' not importable.")

def build_imdb_hf(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("imdb", split="train", cache_dir=root)
    N = len(ds)
    texts_iter = (ds[i]["text"] for i in range(N))
    vocab = build_vocab_from_texts(texts_iter, total=N, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        r = ds[i]
        ids = [vocab[t] for t in basic_split(r["text"])]
        ids = _ensure_nonempty_ids(ids, vocab)[:max_seq_len]
        pairs.append((ids, int(r["label"])))
    return pairs, vocab, 2, len(pairs), "IMDB"

def build_sst2_hf(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("glue", "sst2", split="train", cache_dir=root)
    N = len(ds)
    texts_iter = (ds[i]["sentence"] for i in range(N))
    vocab = build_vocab_from_texts(texts_iter, total=N, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        r = ds[i]
        ids = [vocab[t] for t in basic_split(r["sentence"])]
        ids = _ensure_nonempty_ids(ids, vocab)[:max_seq_len]
        pairs.append((ids, int(r["label"])))
    return pairs, vocab, 2, len(pairs), "SST2"

def build_text_cls_dataset(pairs, vocab, max_seq_len:int, num_classes:int):
    class _ClsDS(torch.utils.data.Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, idx): return pairs[idx]
        @staticmethod
        def collate(batch, pad_idx=vocab[PAD]):
            seqs, labels = zip(*batch)
            lens = [min(len(s), max_seq_len) for s in seqs]
            L = torch.tensor(lens, dtype=torch.long)
            X = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            for i, s in enumerate(seqs):
                s = s[:max_seq_len] if len(s) > 0 else [pad_idx]
                Ls = min(len(s), max_seq_len)
                X[i,:Ls] = torch.tensor(s[:Ls], dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.long)
            return X, L, y
    ds = _ClsDS()
    ds.collate_fn = _ClsDS.collate
    return ds

def build_wmt14_hf(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("wmt14", "de-en", split="train", cache_dir=root)
    N = len(ds)
    srcs = [ds[i]["translation"]["de"] for i in range(N)]
    tgts = [ds[i]["translation"]["en"] for i in range(N)]
    vocab = build_vocab_from_texts(srcs + tgts, total=2*N, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        de = [vocab[t] for t in basic_split(srcs[i])]
        en = [vocab[t] for t in basic_split(tgts[i])]
        de = [vocab[BOS]] + _ensure_nonempty_ids(de, vocab) + [vocab[EOS]]
        en = [vocab[BOS]] + _ensure_nonempty_ids(en, vocab) + [vocab[EOS]]
        pairs.append((de[:max_seq_len], en[:max_seq_len]))
    return pairs, vocab, len(pairs), "WMT14_EN_DE"

def build_cnndm_hf(root, max_seq_len:int, vocab_size:int):
    _hf_require()
    ds = hfds.load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=root)
    N = len(ds)
    srcs = []
    tgts = []
    for i in range(N):
        ex = ds[i]
        srcs.append(ex.get("article") or ex.get("document") or "")
        tgts.append(ex.get("highlights") or ex.get("summary") or "")
    vocab = build_vocab_from_texts(srcs + tgts, total=2*N, max_tokens=vocab_size)
    pairs=[]
    for i in range(N):
        x = [vocab[t] for t in basic_split(srcs[i])]
        y = [vocab[t] for t in basic_split(tgts[i])]
        x = [vocab[BOS]] + _ensure_nonempty_ids(x, vocab) + [vocab[EOS]]
        y = [vocab[BOS]] + _ensure_nonempty_ids(y, vocab) + [vocab[EOS]]
        pairs.append((x[:max_seq_len], y[:max_seq_len]))
    return pairs, vocab, len(pairs), "CNNDM_SUM"

def build_s2s_dataset(pairs, vocab, max_seq_len:int):
    class _S2S(torch.utils.data.Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, idx): return pairs[idx]
        @staticmethod
        def collate(batch, pad_idx=vocab[PAD]):
            srcs, tgts = zip(*batch)
            len_src = [min(len(s), max_seq_len) for s in srcs]
            len_tgt = [min(len(t), max_seq_len) for t in tgts]
            Ls = torch.tensor(len_src, dtype=torch.long)
            Lt = torch.tensor(len_tgt, dtype=torch.long)
            Xs = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            Xt = torch.full((len(batch), max_seq_len), pad_idx, dtype=torch.long)
            for i,(s,t) in enumerate(zip(srcs,tgts)):
                s = s[:max_seq_len] if len(s)>0 else [pad_idx]
                t = t[:max_seq_len] if len(t)>0 else [pad_idx]
                Xs[i,:len(s)] = torch.tensor(s, dtype=torch.long)
                Xt[i,:len(t)] = torch.tensor(t, dtype=torch.long)
            return Xs, Ls, Xt, Lt
    ds = _S2S()
    ds.collate_fn = _S2S.collate
    return ds

def build_vision_datasets(root, name:str, img_size:int=224):
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision not available for vision datasets.")
    tfm_train = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
    ])
    if name == "CIFAR100":
        ds = torchvision.datasets.CIFAR100(root=os.path.join(root,"cifar100"), train=True, download=True, transform=tfm_train)
        num_classes = 100
    elif name == "STL10":
        ds = torchvision.datasets.STL10(root=os.path.join(root,"stl10"), split="train", download=True, transform=tfm_train)
        num_classes = 10
    elif name == "TinyImageNet":
        if _HAS_HF:
            d = hfds.load_dataset("Maysee/tiny-imagenet", split="train", cache_dir=root)
            class TinyDS(torch.utils.data.Dataset):
                def __len__(self): return len(d)
                def __getitem__(self, idx):
                    ex = d[idx]
                    img = ex["image"].convert("RGB")
                    return tfm_train(img), int(ex["label"])
            ds = TinyDS(); num_classes = 200
        else:
            raise RuntimeError("TinyImageNet requires HF datasets (Maysee/tiny-imagenet).")
    else:
        raise RuntimeError(f"Unknown vision dataset {name}")
    return ds, num_classes, len(ds)

class DistilTextClassifier(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=256, nhead:int=8, depth:int=6, dim_ff:int=1024, num_classes:int=2, pad_idx:int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.pad_idx = pad_idx

    def forward(self, x, lengths):
        attn_mask = (x != self.pad_idx)
        h = self.embedding(x)
        h = self.encoder(h, src_key_padding_mask=~attn_mask)
        h = self.norm(h)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(attn_mask.sum(dim=1, keepdim=True), min=1).to(h.dtype)
        return self.fc(pooled)

class VanillaSeq2Seq(nn.Module):
    def __init__(self, vocab_size:int, d_model:int=512, nhead:int=8, depth:int=6, dim_ff:int=2048, pad_idx:int=0):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=depth, num_decoder_layers=depth,
                                          dim_feedforward=dim_ff, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, src, src_len, tgt, tgt_len):
        src_key_padding_mask = (src == self.pad_idx)
        Tt = int(tgt_len.max().item())
        tgt = tgt[:, :Tt]
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Tt).to(src.device)

        hs = self.src_embed(src)
        ht = self.tgt_embed(tgt)
        out = self.transformer(hs, ht,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, Tt

def build_deit_tiny(num_classes:int):
    if not _HAS_TIMM:
        raise RuntimeError("timm is required for DeiT/ViT.")
    return timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

def build_vit_base(num_classes:int):
    if not _HAS_TIMM:
        raise RuntimeError("timm is required for DeiT/ViT.")
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

class _Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-30, beta2=0.999, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]; eps = group["eps"]; beta2 = group["beta2"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if wd != 0: g = g.add(p, alpha=wd)
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                v = state["v"]
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                upd = g / (v.sqrt() + eps)
                p.add_(upd, alpha=-lr)
        return loss

def build_optimizer(name:str, params, lr:float, weight_decay:float=0.0):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9,0.999), eps=1e-8)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9,0.999), eps=1e-8)
    elif name == "adafactor":
        return _Adafactor(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name}")

def build_loader(ds, batch_size:int, num_workers:int=2) -> DataLoader:
    if platform.system() == "Windows": num_workers = 0
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=(DEVICE.type=='cuda'), persistent_workers=False,
                      collate_fn=getattr(ds, "collate_fn", None))

def criterion_for_task(task:str, ignore_index:int=-100):
    if task in ("text_cls","vision_cls"):
        return nn.CrossEntropyLoss()
    elif task == "seq2seq":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError(task)

def measure_batch_timing(model: nn.Module,
                         data_loader: DataLoader,
                         warmup_batches: int,
                         measure_batches: int,
                         lr: float,
                         optimizer_name: str,
                         amp: bool,
                         task: str,
                         pad_idx: Optional[int] = None) -> Dict[str, Any]:

    model = model.to(DEVICE)
    crit = criterion_for_task(task, ignore_index=(pad_idx if pad_idx is not None else -100))
    opt = build_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=0.0)
    scaler = _GradScaler(enabled=(amp and DEVICE.type=='cuda'))

    total_batches = warmup_batches + measure_batches
    it = iter(data_loader)
    model.train()
    total_step_ms=[]; loader_ms_list=[]; compute_ms_list=[]
    ex_count=0; tok_count=0; measured_time_s=0.0

    for i in range(total_batches):
        t0_load = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader); batch = next(it)
        t1_load = time.perf_counter()
        loader_ms = (t1_load - t0_load) * 1000.0

        opt.zero_grad(set_to_none=True)

        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t0_compute = time.perf_counter()

        try:
            if scaler.is_enabled():
                with _amp_autocast(True):
                    if task == "text_cls":
                        X, L, y = batch
                        X = X.to(DEVICE, non_blocking=True)
                        L = L.to(DEVICE, non_blocking=True)
                        y = y.to(DEVICE, non_blocking=True)
                        logits = model(X, L)
                        loss = crit(logits, y)
                        if i >= warmup_batches:
                            ex_count += X.size(0); tok_count += int(L.sum().item())
                    elif task == "vision_cls":
                        X, y = batch
                        X = X.to(DEVICE, non_blocking=True)
                        y = y.to(DEVICE, non_blocking=True)
                        logits = model(X)
                        loss = crit(logits, y)
                        if i >= warmup_batches:
                            ex_count += X.size(0)
                    else:  # seq2seq
                        Xs, Ls, Xt, Lt = batch
                        Xs = Xs.to(DEVICE, non_blocking=True)
                        Xt = Xt.to(DEVICE, non_blocking=True)
                        Ls = Ls.to(DEVICE, non_blocking=True)
                        Lt = Lt.to(DEVICE, non_blocking=True)
                        logits, Tt = model(Xs, Ls, Xt, Lt)
                        loss = crit(logits.reshape(-1, logits.size(-1)), Xt[:, :Tt].reshape(-1))
                        if i >= warmup_batches:
                            ex_count += Xs.size(0)
                            tok_count += int(Lt.clamp(max=Tt).sum().item())
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                if task == "text_cls":
                    X, L, y = batch
                    X = X.to(DEVICE); L = L.to(DEVICE); y = y.to(DEVICE)
                    loss = crit(model(X, L), y)
                    if i >= warmup_batches:
                        ex_count += X.size(0); tok_count += int(L.sum().item())
                elif task == "vision_cls":
                    X, y = batch
                    X = X.to(DEVICE); y = y.to(DEVICE)
                    loss = crit(model(X), y)
                    if i >= warmup_batches:
                        ex_count += X.size(0)
                else:
                    Xs, Ls, Xt, Lt = batch
                    Xs = Xs.to(DEVICE); Xt = Xt.to(DEVICE); Ls = Ls.to(DEVICE); Lt = Lt.to(DEVICE)
                    logits, Tt = model(Xs, Ls, Xt, Lt)
                    loss = crit(logits.reshape(-1, logits.size(-1)), Xt[:, :Tt].reshape(-1))
                    if i >= warmup_batches:
                        ex_count += Xs.size(0)
                        tok_count += int(Lt.clamp(max=Tt).sum().item())
                loss.backward(); opt.step()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                try: del loss
                except: pass
                if DEVICE.type=='cuda':
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                if DEVICE.type=='cuda': torch.cuda.synchronize()
                t1_compute = time.perf_counter()
                if i >= warmup_batches:
                    measured_time_s += (t1_compute - t0_compute)
                continue
            else:
                raise

        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t1_compute = time.perf_counter()

        compute_ms = (t1_compute - t0_compute) * 1000.0
        total_ms = loader_ms + compute_ms
        if i >= warmup_batches:
            total_step_ms.append(total_ms)
            loader_ms_list.append(loader_ms)
            compute_ms_list.append(compute_ms)
            measured_time_s += (t1_compute - t0_compute)

    avg_ms = float(np.mean(total_step_ms)) if total_step_ms else float('nan')
    p90_ms = float(np.percentile(total_step_ms, 90)) if total_step_ms else float('nan')
    avg_loader_ms = float(np.mean(loader_ms_list)) if loader_ms_list else float('nan')
    avg_compute_ms = float(np.mean(compute_ms_list)) if compute_ms_list else float('nan')
    loader_ratio = float(avg_loader_ms / max(1e-9, (avg_loader_ms + avg_compute_ms))) if np.isfinite(avg_loader_ms) else float('nan')

    if measured_time_s > 0:
        examples_per_sec = ex_count / measured_time_s
        tokens_per_sec = (tok_count / measured_time_s) if tok_count > 0 else float('nan')
    else:
        examples_per_sec = float('nan'); tokens_per_sec = float('nan')

    return dict(
        avg_batch_time_ms=avg_ms, p90_batch_time_ms=p90_ms, avg_loader_ms=avg_loader_ms,
        avg_compute_ms=avg_compute_ms, loader_ratio=loader_ratio,
        examples_per_sec=examples_per_sec, tokens_per_sec=tokens_per_sec,
        amp_enabled=bool(scaler.is_enabled())
    )

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mem_bytes_for_params(param_cnt:int) -> int:
    return param_cnt * 4

def optimizer_state_multiplier(name:str) -> int:
    name = name.lower()
    if name in ("adam","adamw","adafactor"):
        return 2
    return 1

def flops_transformer_block(seq_len:int, d_model:int, n_heads:int, d_ff:int):
    d_head = max(1, d_model // n_heads)
    attn_qkv = 3 * d_model * d_model
    attn_scores = seq_len * d_model * d_head
    attn_out = d_model * d_model
    ffn = 2 * d_model * d_ff
    per_token = attn_qkv + attn_out + (2 * attn_scores) + ffn
    return per_token

def main():
    ap = argparse.ArgumentParser("Transformer Benchmark (Per-Batch Timing)")
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--logdir', type=str, default='./logs')

    ap.add_argument('--warmup-batches', type=int, default=5)
    ap.add_argument('--measure-batches', type=int, default=20)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--precision', type=int, default=16, choices=[16,32])
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)

    ap.add_argument('--vocab-size', type=int, default=30000)
    ap.add_argument('--seq-len-txt', type=int, default=128)
    ap.add_argument('--seq-len-wmt', type=int, default=128)
    ap.add_argument('--seq-len-cnndm-src', type=int, default=512)
    ap.add_argument('--seq-len-cnndm-tgt', type=int, default=128)

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)
    gname = gpu_name()
    out_csv = os.path.join(os.path.abspath(args.logdir), f"Transformer_{_safe(gname) if DEVICE.type=='cuda' else 'CPU'}.csv")

    dataset_builders = [
        ("SST2", lambda: build_sst2_hf(args.data_root, args.seq_len_txt, args.vocab_size)),
        ("IMDB", lambda: build_imdb_hf(args.data_root, args.seq_len_txt, args.vocab_size)),
        ("WMT14_EN_DE", lambda: build_wmt14_hf(args.data_root, args.seq_len_wmt, args.vocab_size)),
        ("CNNDM_SUM", lambda: build_cnndm_hf(args.data_root, max(args.seq_len_cnndm_src, args.seq_len_cnndm_tgt), args.vocab_size)),
        ("CIFAR100", lambda: build_vision_datasets(args.data_root, "CIFAR100", img_size=224)),
        ("STL10", lambda: build_vision_datasets(args.data_root, "STL10", img_size=224)),
        ("TinyImageNet", lambda: build_vision_datasets(args.data_root, "TinyImageNet", img_size=224)),
    ]

    built = {}
    for name, fn in dataset_builders:
        bar = tbar(total=1, desc=f"Preparing {name}", leave=True)
        try:
            if name in ("SST2","IMDB"):
                pairs, vocab, nc, N, label = fn()
                ds = build_text_cls_dataset(pairs, vocab, args.seq_len_txt, nc)
                built[name] = dict(ds=ds, vocab=vocab, nc=nc, N=N, label=label)
            elif name in ("WMT14_EN_DE","CNNDM_SUM"):
                pairs, vocab, N, label = fn()
                max_len = args.seq_len_wmt if name=="WMT14_EN_DE" else max(args.seq_len_cnndm_src, args.seq_len_cnndm_tgt)
                ds = build_s2s_dataset(pairs, vocab, max_len)
                built[name] = dict(ds=ds, vocab=vocab, nc=None, N=N, label=label)
            else:  # vision
                ds, nc, N = fn()
                built[name] = dict(ds=ds, vocab=None, nc=nc, N=N, label=name)
        except Exception as e:
            tqdm.write(f"[WARN] {name} unavailable: {e}")
            built[name] = None
        finally:
            bar.update(1); bar.close()

    sst = built.get("SST2")
    imdb = built.get("IMDB")
    wmt = built.get("WMT14_EN_DE")
    cnndm = built.get("CNNDM_SUM")
    cifar = built.get("CIFAR100")
    stl = built.get("STL10")
    tiny = built.get("TinyImageNet")

    sweeps = {
        ("DistilBERT", "SST2"): dict(task="text_cls", ds=sst["ds"] if sst else None, ds_label="SST2",
                                      batch=[8,16,32], lr=[1e-4,2e-4,5e-4], opt=["AdamW","Adafactor"],
                                      seq_len=args.seq_len_txt, vocab=(sst["vocab"] if sst else None),
                                      num_classes=(sst["nc"] if sst else 2), N=(sst["N"] if sst else 0)),
        ("DistilBERT", "IMDB"): dict(task="text_cls", ds=imdb["ds"] if imdb else None, ds_label="IMDB",
                                      batch=[8,16,32], lr=[1e-4,2e-4,5e-4], opt=["AdamW","Adafactor"],
                                      seq_len=args.seq_len_txt, vocab=(imdb["vocab"] if imdb else None),
                                      num_classes=(imdb["nc"] if imdb else 2), N=(imdb["N"] if imdb else 0)),

        ("DeiT-Tiny", "CIFAR100"): dict(task="vision_cls", ds=cifar["ds"] if cifar else None, ds_label="CIFAR100",
                                         batch=[32,48,64], lr=[1e-3,2e-3,3e-3], opt=["AdamW","Adam"],
                                         num_classes=(cifar["nc"] if cifar else 100), N=(cifar["N"] if cifar else 0)),
        ("DeiT-Tiny", "TinyImageNet"): dict(task="vision_cls", ds=tiny["ds"] if tiny else None, ds_label="TinyImageNet",
                                            batch=[32,48,64], lr=[1e-3,2e-3,3e-3], opt=["AdamW","Adam"],
                                            num_classes=(tiny["nc"] if tiny else 200), N=(tiny["N"] if tiny else 0)),

        ("ViT", "CIFAR100"): dict(task="vision_cls", ds=cifar["ds"] if cifar else None, ds_label="CIFAR100",
                                  batch=[4,8,16], lr=[5e-4,1e-3,2e-3], opt=["AdamW","Adam"],
                                  num_classes=(cifar["nc"] if cifar else 100), N=(cifar["N"] if cifar else 0)),
        ("ViT", "STL10"): dict(task="vision_cls", ds=stl["ds"] if stl else None, ds_label="STL10",
                               batch=[4,8,16], lr=[5e-4,1e-3,2e-3], opt=["AdamW","Adam"],
                               num_classes=(stl["nc"] if stl else 10), N=(stl["N"] if stl else 0)),

        ("Transformer", "WMT14_EN_DE"): dict(task="seq2seq", ds=wmt["ds"] if wmt else None, ds_label="WMT14_EN_DE",
                                             batch=[8,16,32], lr=[3e-4,5e-4,1e-3], opt=["AdamW","Adafactor"],
                                             seq_len=args.seq_len_wmt, vocab=(wmt["vocab"] if wmt else None), N=(wmt["N"] if wmt else 0)),
        ("Transformer", "CNNDM_SUM"): dict(task="seq2seq", ds=cnndm["ds"] if cnndm else None, ds_label="CNNDM_SUM",
                                           batch=[2,4,8], lr=[2e-4,3e-4,5e-4], opt=["AdamW","Adafactor"],
                                           seq_len=max(args.seq_len_cnndm_src, args.seq_len_cnndm_tgt),
                                           vocab=(cnndm["vocab"] if cnndm else None), N=(cnndm["N"] if cnndm else 0)),
    }

    rows = []
    for (model_name, ds_name), cfg in sweeps.items():
        if cfg["ds"] is None:
            tqdm.write(f"[WARN] Skipping {model_name} on {ds_name}: dataset not available.")
            continue

        task = cfg["task"]
        is_text = (task == "text_cls")
        is_vision = (task == "vision_cls")
        is_s2s = (task == "seq2seq")

        loader_cache: Dict[int, DataLoader] = {}
        for B in cfg["batch"]:
            try:
                loader_cache[B] = build_loader(cfg["ds"], batch_size=B, num_workers=args.num_workers)
            except RuntimeError as e:
                tqdm.write(f"[ERROR] DataLoader failed for {ds_name} B={B}: {e}")

        combos = list(product(cfg["opt"], cfg["lr"], cfg["batch"]))
        inner_bar = tbar(total=len(combos), desc=f"{model_name} on {cfg['ds_label']}", leave=True)

        for opt_name, lr, B in combos:
            if B not in loader_cache:
                inner_bar.update(1)
                continue
            loader = loader_cache[B]

            if is_text:
                vocab = cfg["vocab"]; pad_idx = vocab[PAD]
                model = DistilTextClassifier(vocab_size=len(vocab), d_model=256, nhead=8, depth=6, dim_ff=1024, num_classes=cfg["num_classes"], pad_idx=pad_idx)
                d_model, n_heads, d_ff, depth = 256, 8, 1024, 6
                arch_family = "encoder_only"
                seq_len = cfg["seq_len"]
                src_len = seq_len; tgt_len = float('nan')
                padding_ratio = float('nan')
                tokens_per_seq = seq_len
                vocab_size = len(vocab)
                d_head = d_model // n_heads
                attention_pattern = "dense"
                norm_style, norm_impl = "preln", "LayerNorm"
                activation = "gelu"

            elif is_vision:
                pad_idx = None
                if model_name == "DeiT-Tiny":
                    model = build_deit_tiny(num_classes=cfg["num_classes"])
                    d_model, n_heads, d_ff, depth = 192, 3, 768, 12
                else:  # ViT Base/16
                    model = build_vit_base(num_classes=cfg["num_classes"])
                    d_model, n_heads, d_ff, depth = 768, 12, 3072, 12
                arch_family = "encoder_only"
                seq_len = 196  # 224x224 / 16x16 patches
                src_len = seq_len; tgt_len = float('nan')
                vocab_size = float('nan')
                d_head = d_model // n_heads
                attention_pattern = "dense"
                norm_style, norm_impl = "preln", "LayerNorm"
                activation = "gelu"
                padding_ratio = float('nan')
                tokens_per_seq = seq_len

            else:  # seq2seq
                vocab = cfg["vocab"]; pad_idx = vocab[PAD]
                model = VanillaSeq2Seq(vocab_size=len(vocab), d_model=512, nhead=8, depth=6, dim_ff=2048, pad_idx=pad_idx)
                d_model, n_heads, d_ff, depth = 512, 8, 2048, 12  # enc+dec
                arch_family = "encoder_decoder"
                seq_len = cfg["seq_len"]
                src_len = seq_len; tgt_len = seq_len
                padding_ratio = float('nan')
                tokens_per_seq = tgt_len
                vocab_size = len(vocab)
                d_head = d_model // n_heads
                attention_pattern = "dense"
                norm_style, norm_impl = "preln", "LayerNorm"
                activation = "gelu"

            model = model.to(DEVICE)
            use_amp = (args.precision==16 and DEVICE.type=='cuda')

            # Params/memory proxies
            P = param_count(model)
            param_bytes = mem_bytes_for_params(P)
            opt_mult = optimizer_state_multiplier(opt_name)
            optimizer_state_bytes = param_bytes * opt_mult
            master_param_bytes = param_bytes if (use_amp) else 0
            total_state_bytes = param_bytes + optimizer_state_bytes + master_param_bytes

            flops_block = flops_transformer_block(int(seq_len if not math.isnan(seq_len) else 128), d_model, n_heads, int(d_ff))
            flops_fwd_per_token = float(flops_block)
            train_mult = 3.0
            flops_train_per_seq = flops_block * tokens_per_seq * depth * train_mult
            flops_train_per_batch = flops_train_per_seq * B
            bytes_per_scalar = 2 if use_amp else 4
            act_bytes_per_seq_proxy = int(tokens_per_seq * (d_model + d_ff) * bytes_per_scalar)
            arithmetic_intensity_train = float(flops_train_per_seq / max(1.0, (param_bytes + act_bytes_per_seq_proxy)))

            try:
                out = measure_batch_timing(
                    model=model, data_loader=loader,
                    warmup_batches=args.warmup_batches, measure_batches=args.measure_batches,
                    lr=lr, optimizer_name=opt_name, amp=use_amp,
                    task=task, pad_idx=pad_idx
                )
            except RuntimeError as e:
                tqdm.write(f"[ERROR] Training failed [{model_name}] on {ds_name} (B={B}, lr={lr}, opt={opt_name}): {e}")
                del model
                if DEVICE.type=='cuda':
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                inner_bar.update(1)
                continue

            if is_vision:
                img_size = 224; patch_size = 16; in_ch = 3
                vs = np.nan
            else:
                img_size = np.nan; patch_size = np.nan; in_ch = np.nan
                vs = int(vocab_size)

            row = {
                'dataset': cfg["ds_label"],
                'dataset_size': int(cfg["N"]),
                'seq_len': float(seq_len) if not is_s2s else float('nan'),
                'src_len': float(src_len) if (is_text or is_vision) else float(src_len),
                'tgt_len': float(tgt_len) if is_s2s else float('nan'),

                'learning_rate': float(lr),
                'batch_size': int(B),
                'effective_batch_size': int(B),
                'precision': int(args.precision),
                'amp_enabled': bool(out['amp_enabled']),
                'optimizer': opt_name,

                'model': model_name,
                'architecture': 'Transformer',
                'arch_family': arch_family,
                'depth': int(depth),
                'n_layers_enc': int(depth//2) if is_s2s else np.nan,
                'n_layers_dec': int(depth//2) if is_s2s else np.nan,
                'd_model': int(d_model),
                'n_heads': int(n_heads),
                'd_head': int(d_head),
                'd_ff': int(d_ff),
                'ffn_multiplier': float(d_ff/float(d_model)),
                'activation': activation,
                'norm_style': norm_style,
                'norm_impl': norm_impl,
                'positional_encoding': 'learned',
                'attention_pattern': attention_pattern,

                'vocab_size': vs,
                'img_size': img_size,
                'patch_size': patch_size,
                'in_channels': in_ch,

                'param_count': int(P),
                'param_bytes': int(param_bytes),
                'optimizer_state_bytes': int(optimizer_state_bytes),
                'master_param_bytes': int(master_param_bytes),
                'total_state_bytes': int(total_state_bytes),

                'flops_fwd_per_token': float(flops_fwd_per_token),
                'flops_train_per_seq': float(flops_train_per_seq),
                'flops_train_per_batch': float(flops_train_per_batch),
                'act_bytes_per_seq_proxy': int(act_bytes_per_seq_proxy),
                'arithmetic_intensity_train': float(arithmetic_intensity_train),

                'avg_batch_time_ms': float(out['avg_batch_time_ms']),
                'p90_batch_time_ms': float(out['p90_batch_time_ms']),
                'loader_wait_ms': float(out['avg_loader_ms']),
                'compute_ms': float(out['avg_compute_ms']),
                'loader_ratio': float(out['loader_ratio']),
                'examples_per_sec': float(out['examples_per_sec']),
                'tokens_per_sec': float(out['tokens_per_sec']),
            }
            row['gpu_name'] = gname
            rows.append(row)

            del model
            if DEVICE.type=='cuda':
                torch.cuda.synchronize(); torch.cuda.empty_cache()

            inner_bar.update(1)

        inner_bar.close()

    if not rows:
        print("[WARN] No rows produced. Check dataset availability and dependencies (timm/torchvision/datasets).")

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != 'gpu_name'] + ['gpu_name']
    df = df[cols]

    cols_to_drop = [
        "padding_ratio","grad_accum_steps","betas_0","betas_1","eps","weight_decay",
        "grad_clip_norm","dropout_p","label_smoothing","scheduler_type","warmup_steps",
        "qkv_fused","kv_shared","weight_tying","parameter_sharing","window_size",
        "fused_bias_gelu","fused_layernorm","fused_optimizer",
        "activation_checkpointing","torch_compile","compile_backend"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved Transformer benchmark to:\n  {out_csv}")

if __name__ == "__main__":
    main()
