#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract residue-aligned PLM token representations.

Special tokens are removed, global token indices map directly to sequence and
one-based residue coordinates, and per-sequence token counts are validated before
writing offsets and layer arrays.
"""

import os, re, json, math, random, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset

# ---------------- utils ----------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_seq(s: str) -> str:
    s = "".join(str(s).upper().split())
    return "".join(ch if "A" <= ch <= "Z" else "X" for ch in s)


def parse_layer_ids(layer_csv: str) -> list[int]:
    return sorted({int(x) for x in str(layer_csv).split(",") if str(x).strip() != ""})


def resolve_output_dtypes(out_dtype: str):
    name = str(out_dtype or "float32").strip().lower()
    if name in {"float32", "fp32"}:
        return np.dtype(np.float32), torch.float32, "float32"
    if name in {"float16", "fp16", "half"}:
        return np.dtype(np.float16), torch.float16, "float16"
    if name in {"bfloat16", "bf16"}:
        try:
            return np.dtype("bfloat16"), torch.bfloat16, "bfloat16"
        except TypeError:
            try:
                import ml_dtypes  # type: ignore

                return (
                    np.dtype(ml_dtypes.bfloat16),
                    torch.bfloat16,
                    "bfloat16(ml_dtypes)",
                )
            except Exception as exc:
                raise ValueError(
                    "--out-dtype=bfloat16 requires NumPy bfloat16 support or ml_dtypes."
                ) from exc
    raise ValueError(
        f"Unsupported --out-dtype:  {out_dtype}(choose float32, float16 or bfloat16)"
    )


def build_hf_items(seqs, max_len, bucket_sort=True):
    items = [
        (i, s, min(len(s), max_len) if isinstance(max_len, int) else len(s))
        for i, s in enumerate(seqs)
    ]
    if bucket_sort:
        items.sort(key=lambda x: x[2], reverse=True)
    return items


class HFSeqDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq_idx, seq, _ = self.items[idx]
        return int(seq_idx), seq


class HFTokenizerCollator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.all_special = set(getattr(tokenizer, "all_special_ids", []) or [])

    def __call__(self, batch):
        idxs = [int(x[0]) for x in batch]
        seqs = [x[1] for x in batch]
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        attn = enc["attention_mask"].bool()
        stm = enc.get("special_tokens_mask", None)
        if stm is None:
            input_ids = enc["input_ids"]
            stm = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in self.all_special:
                stm |= input_ids == sid
        else:
            stm = stm.bool()
        keep = attn & (~stm)
        return torch.as_tensor(idxs, dtype=torch.long), enc, keep


def read_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower_map = {c.lower(): c for c in df.columns}
    need = ["prot_id", "seq", "label"]
    miss = [n for n in need if n not in lower_map]
    if miss:
        raise ValueError(
            f"CSV requires case-insensitive prot_id, seq and label columns. Missing: {miss}"
        )
    df = df[[lower_map["prot_id"], lower_map["seq"], lower_map["label"]]].copy()
    df.columns = ["prot_id", "seq", "label"]
    df = df.dropna(subset=["prot_id", "seq", "label"])
    df["prot_id"] = df["prot_id"].astype(str)
    df["seq"] = df["seq"].map(clean_seq)
    df["label"] = df["label"].astype(str)
    df = df[df["seq"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("Input data are empty.")
    return df


def stratified_sample(df: pd.DataFrame, frac=1.0, seed=42) -> pd.DataFrame:
    if frac >= 1.0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby("label", sort=False):
        k = max(1, int(math.ceil(len(g) * frac)))
        parts.append(g.sample(n=k, random_state=rng))
    out = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=rng)
        .reset_index(drop=True)
    )
    return out


def filter_min_samples(df: pd.DataFrame, min_samples: int):
    cnt = df["label"].value_counts()
    keep = cnt[cnt >= min_samples].index
    df2 = df[df["label"].isin(keep)].reset_index(drop=True)
    return df2, cnt.to_dict(), df2["label"].value_counts().to_dict()


# ---------------- model routing (HF vs ESMC/ESM3) ----------------


def is_esm3_model_name(model_name: str) -> bool:
    if model_name is None:
        return False
    m = model_name.strip().lower()
    return m.startswith("esm3") or "/esm3-" in m


def looks_like_hf_dir(path: str) -> bool:
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "config.json"))
    )


def should_try_hf(model_name: str) -> bool:
    known_esmc = {"esmc_300m", "esmc_600m", "esmc_6b"}
    if looks_like_hf_dir(model_name):
        return True
    if model_name in known_esmc:
        return False
    if is_esm3_model_name(model_name):
        return False
    return "/" in (model_name or "")


def load_hf_model_and_tokenizer(
    model_name_or_dir: str,
    hf_cache: str,
    offline: bool,
    device: str,
    trust_remote_code: bool = True,
    torch_dtype=None,
):
    from transformers import (
        AutoModelForMaskedLM,
        AutoModel,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )

    kw = dict(
        cache_dir=hf_cache,
        local_files_only=offline,
        trust_remote_code=trust_remote_code,
    )

    model = None
    tokenizer = None
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_dir, torch_dtype=torch_dtype, **kw
        )
        tokenizer = getattr(model, "tokenizer", None)
    except Exception:
        model = None

    if model is None:
        model = AutoModel.from_pretrained(
            model_name_or_dir, torch_dtype=torch_dtype, **kw
        )

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, **kw)
        except Exception:
            
            if os.path.isdir(model_name_or_dir):
                tok_json = os.path.join(model_name_or_dir, "tokenizer.json")
                if not os.path.isfile(tok_json):
                    raise
                specials = {}
                for fname in ("special_tokens_map.json", "tokenizer_config.json"):
                    fpath = os.path.join(model_name_or_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            obj = json.load(open(fpath, "r"))

                            def _val(v):
                                return v.get("content", v) if isinstance(v, dict) else v

                            for k in [
                                "unk_token",
                                "pad_token",
                                "bos_token",
                                "eos_token",
                                "sep_token",
                                "mask_token",
                                "cls_token",
                            ]:
                                if k in obj:
                                    specials[k] = _val(obj[k])
                        except Exception:
                            pass
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tok_json, **{k: v for k, v in specials.items() if v}
                )
            else:
                raise

    model.eval().to(device)
    return model, tokenizer


def load_esmc_or_esm3(model_name: str, hf_cache: str, offline: bool, device: str):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if hf_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache

    if is_esm3_model_name(model_name):
        from esm.models.esm3 import ESM3

        client = ESM3.from_pretrained(model_name).to(device).eval()
        return client, "ESM3"
    else:
        from esm.models.esmc import ESMC

        known = {"esmc_300m", "esmc_600m", "esmc_6b"}
        if model_name not in known:
            raise ValueError(
                f"--model must be one of {sorted(known)}, an ESM3 model or a Hugging Face model. Received: {model_name}"
            )
        client = ESMC.from_pretrained(model_name).to(device).eval()
        return client, "ESMC"


def discover_blocks(core: torch.nn.Module):
    cand = []
    pat = re.compile(
        r"(?:^|\.)(?:encoder|model|trunk|transformer|layers|blocks)\.(?:layers\.)?(\d+)$"
    )
    for name, mod in core.named_modules():
        m = pat.search(name)
        if m:
            cand.append((name, mod, int(m.group(1))))
    if not cand:
        for name, mod in core.named_modules():
            m = re.search(r"(\d+)$", name)
            if m:
                cand.append((name, mod, int(m.group(1))))
    if not cand:
        first10 = [n for n, _ in list(core.named_modules())[:10]]
        raise RuntimeError("No hookable Transformer layers were found; examples: " + ", ".join(first10))
    cand.sort(key=lambda x: x[2])
    uniq, seen = [], set()
    for name, mod, idx in cand:
        if id(mod) not in seen:
            uniq.append((name, mod, idx))
            seen.add(id(mod))
    return uniq





def _infer_hf_layers_and_dim(model, tokenizer, seqs, device, max_len=None):
    test = seqs[0][: (max_len if isinstance(max_len, int) else len(seqs[0]))]
    enc = tokenizer(
        test,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    if hs is None:
        raise RuntimeError("Hugging Face forward pass did not return hidden_states.")
    n_layers = len(hs) - 1
    hidden_dim = hs[1].shape[-1]
    return n_layers, int(hidden_dim)


def compute_token_counts_hf(tokenizer, seqs, max_len, batch_size, bucket_sort=True):
    """
    Lightweight pass one: count residue tokens on CPU without padding or GPU transfer.
    """
    items = build_hf_items(seqs=seqs, max_len=max_len, bucket_sort=bucket_sort)

    all_special = set(getattr(tokenizer, "all_special_ids", []) or [])
    counts = np.empty(len(seqs), dtype=np.int32)
    pbar = tqdm(
        total=math.ceil(len(items) / batch_size), desc="HF pass1: count residue tokens"
    )
    for j in range(0, len(items), batch_size):
        chunk = items[j : j + batch_size]
        idxs = [x[0] for x in chunk]
        sss = [x[1] for x in chunk]
        enc = tokenizer(
            sss,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"]
        stm_list = enc.get("special_tokens_mask", None)
        if stm_list is not None:
            lens = [sum(1 for sp in stm if int(sp) == 0) for stm in stm_list]
        else:
            lens = [
                sum(1 for tid in ids if int(tid) not in all_special)
                for ids in input_ids
            ]
        for k, ii in enumerate(idxs):
            counts[ii] = int(lens[k])
        pbar.update(1)
    pbar.close()
    return counts


@torch.no_grad()
def write_tokens_hf_to_memmap(
    model,
    tokenizer,
    seqs,
    layer_ids,
    tok_len,
    offsets,
    out_mm_by_layer,
    pooled_sum_by_layer,
    out_torch_dtype=torch.float32,
    dataloader_workers=0,
    dataloader_prefetch=2,
    pin_memory=False,
    device="cuda",
    max_len=None,
    batch_size=32,
    bucket_sort=True,
):
    """
    Write residue-token embeddings and verify keep.sum == tok_len[seq_idx].
    """
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    items = build_hf_items(seqs=seqs, max_len=max_len, bucket_sort=bucket_sort)
    ds = HFSeqDataset(items)
    collator = HFTokenizerCollator(tokenizer=tokenizer, max_len=max_len)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(max(0, dataloader_workers)),
        "pin_memory": bool(pin_memory),
        "collate_fn": collator,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(dataloader_prefetch))
        
        loader_kwargs["multiprocessing_context"] = "fork"
    dl = DataLoader(ds, **loader_kwargs)

    pbar = tqdm(
        total=math.ceil(len(items) / batch_size), desc="HF pass2: write residue tokens"
    )
    special_stats = {"n_seq": 0, "sum_total_tokens": 0, "sum_residue_tokens": 0}
    token_layer_ids = sorted(out_mm_by_layer.keys())
    pooled_only_layer_ids = sorted(pooled_sum_by_layer.keys())
    for idx_tensor, enc_full, keep in dl:
        idxs = idx_tensor.tolist()
        write_order = np.argsort(
            np.asarray([offsets[ii] for ii in idxs], dtype=np.int64)
        )
        attn = enc_full["attention_mask"].bool()

        # forward
        enc = {}
        for k, v in enc_full.items():
            if k not in ("input_ids", "attention_mask", "token_type_ids"):
                continue
            enc[k] = v.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out_obj = model(**enc, output_hidden_states=True)
        else:
            out_obj = model(**enc, output_hidden_states=True)

        hs = out_obj.hidden_states  # [emb] + layers
        keep_dev = keep.to(device=device, non_blocking=True)

        for b, seq_idx in enumerate(idxs):
            n_keep = int(keep[b].sum().item())
            expect = int(tok_len[seq_idx])
            if n_keep != expect:
                raise RuntimeError(
                    f"[HF length mismatch] seq_index={seq_idx} expect_tok_len={expect} got_keep={n_keep}. "
                    "Tokenizer residue-token counts do not match the sequence-length and truncation contract."
                )

        
        for lid in token_layer_ids:
            H_gpu = hs[1 + lid]  # [B,T,D]
            for b_ord in write_order:
                b = int(b_ord)
                seq_idx = int(idxs[b])
                n_keep = int(tok_len[seq_idx])
                if n_keep <= 0:
                    continue
                start = int(offsets[seq_idx])
                end = start + n_keep
                h_sel = H_gpu[b][keep_dev[b]].to(dtype=out_torch_dtype)
                h_sel_cpu = h_sel.cpu()
                try:
                    h_np = h_sel_cpu.numpy()
                except TypeError:
                    
                    h_np = (
                        h_sel_cpu.float()
                        .numpy()
                        .astype(out_mm_by_layer[lid].dtype, copy=False)
                    )
                out_mm_by_layer[lid][start:end, :] = h_np

        
        if pooled_only_layer_ids:
            keep_dev_3d = keep_dev.unsqueeze(-1)
            batch_indices = np.asarray(idxs, dtype=np.int64)
            for lid in pooled_only_layer_ids:
                H = hs[1 + lid]  # [B,T,D]
                pooled_sum = (H * keep_dev_3d).sum(dim=1).float().cpu().numpy()  # [B,D]
                pooled_sum_by_layer[lid][batch_indices, :] += pooled_sum

        # stats
        special_stats["n_seq"] += len(idxs)
        special_stats["sum_total_tokens"] += int(attn.sum().item())
        special_stats["sum_residue_tokens"] += int(keep.sum().item())

        del out_obj, hs, enc, enc_full, keep_dev, idx_tensor
        pbar.update(1)

    pbar.close()
    return special_stats





def crop_to_residues_esm(H_1d: np.ndarray, L_eff: int):
    """
    H_1d is a [T, D] NumPy array that may contain BOS/EOS tokens.
    Returns H_res [L_eff, D], prefix_special and suffix_special.

    Rules:
    - T == L_eff: no special tokens.
    - T == L_eff + 2: remove one token from each end.
    - T == L_eff + 1: remove the leading token, which is normally BOS.
    - Larger differences: center-crop to L_eff and record the removed prefix/suffix.
    - T < L_eff: raise an error to prevent offset drift.
    """
    T = int(H_1d.shape[0])
    if T < L_eff:
        raise RuntimeError(
            f"[ESM length mismatch] model_T={T} < L_eff={L_eff}. Residues cannot be aligned safely."
        )

    if T == L_eff:
        return H_1d[:L_eff], 0, 0
    if T == L_eff + 2:
        return H_1d[1 : 1 + L_eff], 1, 1
    if T == L_eff + 1:
        
        return H_1d[1 : 1 + L_eff], 1, 0

    
    extra = T - L_eff
    start = extra // 2
    end = start + L_eff
    if end > T:
        start = max(0, T - L_eff)
        end = start + L_eff
    prefix = start
    suffix = T - end
    return H_1d[start:end], prefix, suffix


@torch.no_grad()
def write_tokens_esm_to_memmap(
    client,
    seqs,
    layer_ids,
    tok_len,
    offsets,
    out_mm_by_layer,
    pooled_sum_by_layer,
    out_torch_dtype=torch.float32,
    device="cuda",
    max_len=None,
):
    """
    Run one sequence at a time, capture layer outputs through hooks and crop to tok_len[i] residues.
    """
    from esm.sdk.api import ESMProtein

    core = getattr(client, "model", client)
    blocks = discover_blocks(core)
    Ltot = len(blocks)
    layer_ids = sorted(list(set([max(0, min(Ltot - 1, int(l))) for l in layer_ids])))

    cache = {lid: [] for lid in layer_ids}
    handles = []

    def mk_hook(lid):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            if h.dim() == 2:
                h = h.unsqueeze(0)
            cache[lid].append(h.detach())

        return hook

    for lid in layer_ids:
        _, mod, _ = blocks[lid]
        handles.append(mod.register_forward_hook(mk_hook(lid)))

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    stats = {
        "n_seq": 0,
        "case_T_eq_L": 0,
        "case_T_eq_L1": 0,
        "case_T_eq_L2": 0,
        "case_other": 0,
        "sum_prefix_special": 0,
        "sum_suffix_special": 0,
    }

    pbar = tqdm(total=len(seqs), desc="ESM pass2: write residue tokens (per-seq)")
    for i, s in enumerate(seqs):
        s2 = s[:max_len] if (isinstance(max_len, int) and len(s) > max_len) else s
        L_eff = int(tok_len[i])  
        if L_eff != len(s2):
            
            raise RuntimeError(
                f"[Internal] tok_len[{i}]={L_eff} != len(truncated_seq)={len(s2)}"
            )

        prot = ESMProtein(sequence=s2)
        packed = client.encode(prot) if hasattr(client, "encode") else prot
        if hasattr(packed, "to"):
            packed = packed.to(device)

        def _forward():
            return (
                client.logits(packed) if hasattr(client, "logits") else client(packed)
            )

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = _forward()
        else:
            _ = _forward()

        
        for lid in layer_ids:
            Hs = cache[lid]
            cache[lid] = []
            if not Hs:
                raise RuntimeError(
                    f"[ESM hook missing] seq_index={i} layer={lid} No layer output was captured."
                )
            H = Hs[0]
            if H.size(0) > 1:
                H = H[:1]
            H = H[0].float().cpu().numpy()  # [T,D]
            T = int(H.shape[0])

            H_res, prefix, suffix = crop_to_residues_esm(H, L_eff)
            if H_res.shape[0] != L_eff:
                raise RuntimeError(
                    f"[ESM crop mismatch] seq_index={i} layer={lid} got={H_res.shape[0]} expect={L_eff} (T={T})"
                )

            
            if lid == layer_ids[0]:
                stats["n_seq"] += 1
                if T == L_eff:
                    stats["case_T_eq_L"] += 1
                elif T == L_eff + 1:
                    stats["case_T_eq_L1"] += 1
                elif T == L_eff + 2:
                    stats["case_T_eq_L2"] += 1
                else:
                    stats["case_other"] += 1
                stats["sum_prefix_special"] += int(prefix)
                stats["sum_suffix_special"] += int(suffix)

            if lid in pooled_sum_by_layer:
                pooled_sum_by_layer[lid][i, :] += H_res.sum(axis=0, dtype=np.float32)
            else:
                start = int(offsets[i])
                end = start + L_eff
                if out_mm_by_layer[lid].dtype == np.float32:
                    out_mm_by_layer[lid][start:end, :] = H_res
                else:
                    out_mm_by_layer[lid][start:end, :] = H_res.astype(
                        out_mm_by_layer[lid].dtype, copy=False
                    )

        pbar.update(1)

    pbar.close()

    for h in handles:
        h.remove()

    return stats


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata", required=True, help="CSV with case-insensitive prot_id, seq and label columns"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model",
        default="esmc_600m",
        help="Hugging Face directory/identifier, ESMC model name or ESM3 model name",
    )
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default=None, help="cuda or cpu; selected automatically by default")

    ap.add_argument(
        "--layers", type=str, default="24", help="Comma-separated layers, for example 24 or 24,28,32"
    )
    ap.add_argument(
        "--pool-only-layers",
        type=str,
        default="",
        help="Comma-separated layers saved as sequence means only; must be a subset of --layers",
    )
    ap.add_argument(
        "--out-dtype",
        type=str,
        default="float16",
        help="Token-layer output dtype: float32, float16 or bfloat16",
    )
    ap.add_argument("--max-len", type=int, default=786)
    ap.add_argument("--bs", type=int, default=32, help="Hugging Face batch size; the ESM path processes one sequence at a time")
    ap.add_argument(
        "--bucket-sort", action="store_true", help="Compatibility flag; length bucketing is enabled by default"
    )
    ap.add_argument(
        "--no-bucket-sort", action="store_true", help="Disable length bucketing"
    )
    ap.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin Hugging Face input tensors before asynchronous GPU transfer",
    )
    ap.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Hugging Face pass-two DataLoader workers; zero batches in the main process",
    )
    ap.add_argument(
        "--dataloader-prefetch",
        type=int,
        default=2,
        help="Hugging Face pass-two DataLoader prefetch factor when workers are enabled",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--min-samples-per-class", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- read + filter ----
    df = read_metadata(args.metadata)
    df_s = stratified_sample(df, frac=args.sample_frac, seed=args.seed)
    df_f, cnt_before, cnt_after = filter_min_samples(df_s, args.min_samples_per_class)

    
    df_f = df_f.copy()
    df_f["seq"] = df_f["seq"].astype(str).str.slice(0, int(args.max_len))
    df_f["aa_len"] = df_f["seq"].str.len().astype(int)

    
    json.dump(
        cnt_before,
        open(os.path.join(args.output, "class_counts_before.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump(
        cnt_after,
        open(os.path.join(args.output, "class_counts_after.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    df_s.to_csv(os.path.join(args.output, "sample_metadata_raw.csv"), index=False)

    seqs = df_f["seq"].tolist()
    layer_ids = parse_layer_ids(args.layers)
    pool_only_requested = set(parse_layer_ids(args.pool_only_layers))
    if not pool_only_requested.issubset(set(layer_ids)):
        bad = sorted(pool_only_requested - set(layer_ids))
        raise ValueError(f"--pool-only-layers must be a subset of --layers. Invalid layers: {bad}")

    # ---- load model ----
    model_name = args.model
    use_hf = should_try_hf(model_name) or looks_like_hf_dir(model_name)

    bucket_sort = not bool(args.no_bucket_sort)

    if use_hf:
        torch_dtype = (
            torch.bfloat16
            if (
                device.startswith("cuda")
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            else None
        )
        model, tokenizer = load_hf_model_and_tokenizer(
            model_name_or_dir=model_name,
            hf_cache=args.hf_cache,
            offline=args.offline,
            device=device,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model_type = "HF"
        n_layers, hidden_dim = _infer_hf_layers_and_dim(
            model, tokenizer, seqs, device, args.max_len
        )
        layer_ids = [max(0, min(n_layers - 1, int(l))) for l in layer_ids]
        layer_ids = sorted(set(layer_ids))
        pool_only_layer_ids = sorted(
            [lid for lid in layer_ids if lid in pool_only_requested]
        )
        
        tok_len = compute_token_counts_hf(
            tokenizer=tokenizer,
            seqs=seqs,
            max_len=args.max_len,
            batch_size=args.bs,
            bucket_sort=bucket_sort,
        ).astype(np.int32)
    else:
        client, model_type = load_esmc_or_esm3(
            model_name=model_name,
            hf_cache=args.hf_cache,
            offline=args.offline,
            device=device,
        )
        model = None
        tokenizer = None
        hidden_dim = None

        
        tok_len = df_f["aa_len"].to_numpy(dtype=np.int32, copy=True)

        
        idx0 = 0
        while idx0 < len(seqs) and len(seqs[idx0]) == 0:
            idx0 += 1
        if idx0 >= len(seqs):
            raise RuntimeError("No valid sequence is available to infer hidden_dim.")

        core = getattr(client, "model", client)
        blocks = discover_blocks(core)
        Ltot = len(blocks)
        layer_ids = [max(0, min(Ltot - 1, int(l))) for l in layer_ids]
        layer_ids = sorted(set(layer_ids))
        pool_only_layer_ids = sorted(
            [lid for lid in layer_ids if lid in pool_only_requested]
        )
        probe_lid = layer_ids[0]

        cache = []

        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            cache.append(h.detach())

        _, mod, _ = blocks[probe_lid]
        hdl = mod.register_forward_hook(hook)

        from esm.sdk.api import ESMProtein

        try:
            prot = ESMProtein(sequence=seqs[idx0])
            packed = client.encode(prot) if hasattr(client, "encode") else prot
            if hasattr(packed, "to"):
                packed = packed.to(device)
            _ = client.logits(packed) if hasattr(client, "logits") else client(packed)
            H = cache[0]
            if H.dim() == 2:
                H = H.unsqueeze(0)
            hidden_dim = int(H.shape[-1])
        finally:
            hdl.remove()

    # ---- offsets ----
    offsets = np.zeros(len(tok_len) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(tok_len.astype(np.int64))
    total_tokens = int(offsets[-1])

    
    df_out = df_f.copy()
    df_out["tok_len"] = tok_len.astype(int)
    df_out["tok_start"] = offsets[:-1]
    df_out["tok_end"] = offsets[1:]
    df_out.to_csv(
        os.path.join(args.output, "sample_metadata_filtered.csv"), index=False
    )

    np.save(os.path.join(args.output, "seq_token_offsets.npy"), offsets)
    np.save(os.path.join(args.output, "seq_token_len.npy"), tok_len)

    out_np_dtype, out_torch_dtype, out_dtype_label = resolve_output_dtypes(
        args.out_dtype
    )
    token_layer_ids = [lid for lid in layer_ids if lid not in set(pool_only_layer_ids)]

    # ---- preallocate memmaps ----
    out_tok_dir = os.path.join(args.output, "token_layers")
    os.makedirs(out_tok_dir, exist_ok=True)
    out_pool_dir = os.path.join(args.output, "pooled_layers")
    os.makedirs(out_pool_dir, exist_ok=True)

    mm_by_layer = {}
    for lid in token_layer_ids:
        path = os.path.join(out_tok_dir, f"layer_{lid}.tokens.npy")
        mm_by_layer[lid] = open_memmap(
            path, mode="w+", dtype=out_np_dtype, shape=(total_tokens, int(hidden_dim))
        )
    pooled_sum_by_layer = {
        lid: np.zeros((len(seqs), int(hidden_dim)), dtype=np.float32)
        for lid in pool_only_layer_ids
    }

    summary = {
        "model_type": model_type,
        "model_name": model_name,
        "N_sequences": int(len(seqs)),
        "hidden_dim": int(hidden_dim),
        "layers": [int(x) for x in layer_ids],
        "token_layers_written": [int(x) for x in token_layer_ids],
        "pooled_only_layers": [int(x) for x in pool_only_layer_ids],
        "token_dtype": out_dtype_label,
        "bucket_sort": bool(bucket_sort),
        "pin_memory": bool(args.pin_memory),
        "dataloader_workers": int(max(0, args.dataloader_workers)),
        "dataloader_prefetch": int(max(1, args.dataloader_prefetch)),
        "total_tokens": int(total_tokens),
        "residue_aligned": True,
        "special_tokens_stripped": True,
        "token_files": {
            str(lid): os.path.join("token_layers", f"layer_{lid}.tokens.npy")
            for lid in token_layer_ids
        },
        "pooled_sequence_files": {
            str(lid): os.path.join("pooled_layers", f"layer_{lid}.npy")
            for lid in pool_only_layer_ids
        },
    }
    json.dump(
        summary,
        open(os.path.join(args.output, "token_dump_summary.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )

    # ---- pass2 write ----
    crop_stats = {"model_type": model_type}
    if use_hf:
        st = write_tokens_hf_to_memmap(
            model=model,
            tokenizer=tokenizer,
            seqs=seqs,
            layer_ids=layer_ids,
            tok_len=tok_len,
            offsets=offsets,
            out_mm_by_layer=mm_by_layer,
            pooled_sum_by_layer=pooled_sum_by_layer,
            out_torch_dtype=out_torch_dtype,
            dataloader_workers=int(max(0, args.dataloader_workers)),
            dataloader_prefetch=int(max(1, args.dataloader_prefetch)),
            pin_memory=bool(args.pin_memory),
            device=device,
            max_len=args.max_len,
            batch_size=args.bs,
            bucket_sort=bucket_sort,
        )
        crop_stats.update(st)
    else:
        st = write_tokens_esm_to_memmap(
            client=client,
            seqs=seqs,
            layer_ids=layer_ids,
            tok_len=tok_len,
            offsets=offsets,
            out_mm_by_layer=mm_by_layer,
            pooled_sum_by_layer=pooled_sum_by_layer,
            out_torch_dtype=out_torch_dtype,
            device=device,
            max_len=args.max_len,
        )
        crop_stats.update(st)

    if pooled_sum_by_layer:
        denom = tok_len.astype(np.float32, copy=False).reshape(-1, 1)
        for lid, sum_arr in pooled_sum_by_layer.items():
            pooled = np.divide(
                sum_arr,
                denom,
                out=np.zeros_like(sum_arr, dtype=np.float32),
                where=(denom > 0),
            )
            np.save(
                os.path.join(out_pool_dir, f"layer_{lid}.npy"),
                pooled.astype(np.float32, copy=False),
            )

    json.dump(
        crop_stats,
        open(os.path.join(args.output, "crop_stats.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )

    # flush
    for lid in list(mm_by_layer.keys()):
        del mm_by_layer[lid]

    print(f"[OK] residue-aligned token embeddings saved to: {args.output}")
    print("  - tokens.npy contains residue tokens only")
    print("  - residue_pos_1based is the amino-acid coordinate from 1 to tok_len")
    print("  - crop_stats.json records special-token removal")


if __name__ == "__main__":
    main()
