#!/usr/bin/env python3
"""Extract layer-wise sequence representations from metadata CSV or FASTA input.

The script saves filtered metadata, class counts, run configuration and layer-wise
NumPy arrays using attention-mask mean pooling.
"""

import os, argparse, json, math, random, re, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm



def _force_offline():
    """Set offline environment variables before importing Hugging Face libraries"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



if "--offline" in sys.argv:
    _force_offline()





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_seq(s: str) -> str:
    s = "".join(str(s).upper().split())
    return "".join(ch if "A" <= ch <= "Z" else "X" for ch in s)


def read_metadata(csv_path: str):
    df = pd.read_csv(csv_path)
    lower_map = {c.lower(): c for c in df.columns}
    need = ["prot_id", "seq", "label"]
    if not all(n in lower_map for n in need):
        missing = [n for n in need if n not in lower_map]
        raise ValueError(
            f"CSV requires case-insensitive prot_id, seq and label columns. Missing: {missing}"
        )
    df = df[[lower_map["prot_id"], lower_map["seq"], lower_map["label"]]].copy()
    df.columns = ["prot_id", "seq", "label"]
    df = df.dropna(subset=["prot_id", "seq", "label"])
    df["seq"] = df["seq"].map(clean_seq)
    df["label"] = df["label"].astype(str)
    if df.empty:
        raise ValueError("Input data are empty.")
    return df


def read_fasta_to_df(fa_path: str):
    """Read FASTA input into prot_id, seq and label columns with label set to UNLABELED."""
    import gzip

    if not os.path.isfile(fa_path):
        raise FileNotFoundError(f"FASTA file does not exist: {fa_path}")
    open_fn = gzip.open if fa_path.endswith(".gz") else open
    ids, seqs = [], []
    cur_id, buf = None, []
    with open_fn(fa_path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and buf:
                    seq = clean_seq("".join(buf))
                    if seq:
                        ids.append(cur_id)
                        seqs.append(seq)
                cur_id = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line)
        if cur_id is not None and buf:
            seq = clean_seq("".join(buf))
            if seq:
                ids.append(cur_id)
                seqs.append(seq)
    if not ids:
        raise ValueError(f"FASTA contains no valid sequence: {fa_path}")
    df = pd.DataFrame({"prot_id": ids, "seq": seqs})
    df["label"] = "UNLABELED"
    return df


def stratified_sample(df: pd.DataFrame, frac=1.0, seed=42):
    if frac >= 1.0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    parts = []
    for _lab, g in df.groupby("label", sort=False):
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
    keep_labels = cnt[cnt >= min_samples].index
    df2 = df[df["label"].isin(keep_labels)].reset_index(drop=True)
    return df2, cnt.to_dict(), df2["label"].value_counts().to_dict()





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


# ---------------------------

# ---------------------------


def load_hf_model_and_tokenizer(
    model_name_or_dir: str,
    hf_cache: str,
    offline: bool,
    device: str,
    trust_remote_code: bool = True,
    torch_dtype=None,
):
    """Support custom models that expose a tokenizer attribute"""

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import (
        AutoModelForMaskedLM,
        AutoModel,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )

    load_kwargs = dict(
        cache_dir=hf_cache,
        local_files_only=offline,
        trust_remote_code=trust_remote_code,
    )

    model = None
    tokenizer = None

    
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_dir, torch_dtype=torch_dtype, **load_kwargs
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_name_or_dir, torch_dtype=torch_dtype, **load_kwargs
        )

    
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        print(f"[INFO] loaded tokenizer from model attribute:  {type(tokenizer).__name__}")

    
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, **load_kwargs)
        except Exception as e:
            
            if os.path.isdir(model_name_or_dir):
                tok_json = os.path.join(model_name_or_dir, "tokenizer.json")
                if os.path.isfile(tok_json):
                    
                    pass
            if tokenizer is None:
                raise RuntimeError(f"Unable to load a tokenizer for model {model_name_or_dir} .")

    model.eval().to(device)
    return model, tokenizer


def _batch_iterator_by_length(seqs, batch_size, max_len=None, sort_desc=True):
    items = []
    for i, s in enumerate(seqs):
        eff_len = min(len(s), max_len) if isinstance(max_len, int) else len(s)
        items.append((i, s, eff_len))
    if sort_desc:
        items.sort(key=lambda x: x[2], reverse=True)
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        idxs = [it[0] for it in chunk]
        sss = [it[1] for it in chunk]
        batches.append((idxs, sss))
    return batches


def _masked_mean_pool(H, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(H.dtype)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    pooled = (H * mask).sum(dim=1) / denom
    return pooled


def pool_hidden_states(H, attention_mask, pooling: str = "mean"):
    mean_pooled = _masked_mean_pool(H, attention_mask)
    if pooling == "mean":
        return mean_pooled
    if pooling == "cls+mean":
        cls_pooled = H[:, 0, :]
        return torch.cat([cls_pooled, mean_pooled], dim=-1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


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
    return n_layers, hidden_dim


@torch.no_grad()
def layer_embeddings_hf_batched(
    model,
    tokenizer,
    seqs,
    layer_ids,
    device="cuda",
    max_len=None,
    batch_size=32,
    bucket_sort=True,
    pooling="mean",
):
    n_layers, hidden_dim = _infer_hf_layers_and_dim(
        model, tokenizer, seqs, device, max_len
    )
    layer_ids = sorted(
        list(set([max(0, min(n_layers - 1, int(l))) for l in layer_ids]))
    )
    N = len(seqs)
    pooled_dim = hidden_dim * 2 if pooling == "cls+mean" else hidden_dim
    out = {lid: np.empty((N, pooled_dim), dtype=np.float32) for lid in layer_ids}
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    batches = _batch_iterator_by_length(
        seqs, batch_size=batch_size, max_len=max_len, sort_desc=bucket_sort
    )
    pbar = tqdm(total=len(batches), desc="HF batched probing")
    for idxs, sss in batches:
        enc = tokenizer(
            sss,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
            add_special_tokens=True,
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out_obj = model(**enc, output_hidden_states=True)
        else:
            out_obj = model(**enc, output_hidden_states=True)
        hs = out_obj.hidden_states
        attn = enc["attention_mask"]
        for lid in layer_ids:
            H = hs[1 + lid]
            pooled = pool_hidden_states(H, attn, pooling=pooling)
            arr = pooled.float().cpu().numpy().astype(np.float32, copy=False)
            for j, idx in enumerate(idxs):
                out[lid][idx, :] = arr[j, :]
        del out_obj, hs, enc
        if use_amp and torch.cuda.is_available():
            torch.cuda.synchronize()
        pbar.update(1)
    pbar.close()
    return out


def load_esmc_or_esm3(model_name: str, hf_cache: str, offline: bool, device: str):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if hf_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    if is_esm3_model_name(model_name):
        try:
            from esm.models.esm3 import ESM3
        except ImportError as e:
            raise RuntimeError("esm.models.esm3 is unavailable.") from e
        client = ESM3.from_pretrained(model_name).to(device).eval()
        return client, "ESM3-open"
    else:
        try:
            from esm.models.esmc import ESMC
        except ImportError as e:
            raise RuntimeError("esm.models.esmc is unavailable.") from e
        known = {"esmc_300m", "esmc_600m", "esmc_6b"}
        if model_name not in known:
            raise ValueError(
                f"--model must be one of {sorted(known)} or an ESM3 or Hugging Face model. Received: {model_name}"
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


@torch.no_grad()
def layer_embeddings_esm(client, seqs, layer_ids, device="cuda", max_len=None):
    try:
        from esm.sdk.api import ESMProtein
    except Exception as e:
        raise RuntimeError("The esm package providing esm.sdk.api.ESMProtein is required.") from e
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
    out_buffers = {lid: [] for lid in layer_ids}
    for s in seqs:
        s2 = s[:max_len] if (isinstance(max_len, int) and len(s) > max_len) else s
        prot = ESMProtein(sequence=s2)
        L_i = len(s2)
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
                continue
            H = Hs[0]
            if H.dim() == 2:
                H = H.unsqueeze(0)
            if H.size(0) > 1:
                H = H[:1]
            H = H[0]  # [L,D]
            vec = H[:L_i].mean(0, keepdim=True)
            out_buffers[lid].append(
                vec.float().cpu().numpy().astype(np.float32, copy=False)
            )
    for h in handles:
        h.remove()
    return {
        lid: (
            np.vstack(out_buffers[lid])
            if len(out_buffers[lid]) > 0
            else np.zeros((len(seqs), 1), dtype=np.float32)
        )
        for lid in layer_ids
    }





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="CSV containing prot_id, seq and label")
    ap.add_argument(
        "--fasta",
        default=None,
        help="Unlabelled FASTA input (.fa, .fasta or .faa; optionally gzipped)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model",
        default=None,
        help="Hugging Face directory/identifier, ESMC model name or ESM3 model name",
    )
    ap.add_argument("--client_pth", default=None, help="Optional custom client checkpoint")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default=None, help="cuda or cpu; selected automatically by default")
    ap.add_argument("--layers", type=str, default="0,8,16,24,32,40,47")
    ap.add_argument("--max-len", type=int, default=786)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--bucket-sort", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--min-samples-per-class", type=int, default=10)
    ap.add_argument("--pooling", choices=["mean", "cls+mean"], default="mean")
    args = ap.parse_args()

    
    if (args.metadata is None) and (args.fasta is None):
        raise ValueError("Provide exactly one of --metadata or --fasta.")
    if (args.metadata is not None) and (args.fasta is not None):
        raise ValueError("--metadata and --fasta are mutually exclusive.")

    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    
    use_labels = args.metadata is not None
    if use_labels:
        
        df = read_metadata(args.metadata)
        df_s = stratified_sample(df, frac=args.sample_frac, seed=args.seed)
        df_f, cnt_before, cnt_after = filter_min_samples(
            df_s, args.min_samples_per_class
        )

        
        df_s.to_csv(os.path.join(args.output, "sample_metadata_raw.csv"), index=False)
        df_f.to_csv(
            os.path.join(args.output, "sample_metadata_filtered.csv"), index=False
        )
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

        # if df_f['label'].nunique() < 2:
        
    else:
        
        df = read_fasta_to_df(args.fasta)
        if args.sample_frac >= 1.0:
            df_s = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        else:
            df_s = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(
                drop=True
            )
        df_f = df_s.copy()

        
        cnt_before = {"UNLABELED": int(len(df_s))}
        cnt_after = {"UNLABELED": int(len(df_f))}

        df_s.to_csv(os.path.join(args.output, "sample_metadata_raw.csv"), index=False)
        df_f.to_csv(
            os.path.join(args.output, "sample_metadata_filtered.csv"), index=False
        )
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

    
    run_cfg = {
        "mode": "metadata" if use_labels else "fasta",
        "metadata": args.metadata,
        "fasta": args.fasta,
        "model": args.model or "esmc_600m",
        "client_pth": args.client_pth,
        "layers": args.layers,
        "max_len": args.max_len,
        "bs": args.bs,
        "bucket_sort": bool(args.bucket_sort),
        "seed": args.seed,
        "sample_frac": args.sample_frac,
        "min_samples_per_class": args.min_samples_per_class,
        "pooling": args.pooling,
        "device": device,
    }
    json.dump(
        run_cfg,
        open(os.path.join(args.output, "run_config.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )

    seqs = df_f["seq"].tolist()
    layer_ids = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    
    model_name = args.model or "esmc_600m"
    if args.client_pth:
        if not os.path.isfile(args.client_pth):
            raise FileNotFoundError(f"--client_pth does not exist: {args.client_pth}")
        client = torch.load(args.client_pth, map_location="cpu", weights_only=False)
        client = client.to(device).eval()
        use_hf = False
        model_type = "client_pth"
    else:
        if should_try_hf(model_name) or looks_like_hf_dir(model_name):
            model, tokenizer = load_hf_model_and_tokenizer(
                model_name_or_dir=model_name,
                hf_cache=args.hf_cache,
                offline=args.offline,
                device=device,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=(
                    torch.bfloat16
                    if (
                        device.startswith("cuda")
                        and torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported()
                    )
                    else None
                ),
            )
            use_hf = True
            model_type = "HF"
        else:
            client, model_type = load_esmc_or_esm3(
                model_name=model_name,
                hf_cache=args.hf_cache,
                offline=args.offline,
                device=device,
            )
            use_hf = False

    if use_hf:
        X_by_layer = layer_embeddings_hf_batched(
            model=model,
            tokenizer=tokenizer,
            seqs=seqs,
            layer_ids=layer_ids,
            device=device,
            max_len=args.max_len,
            batch_size=args.bs,
            bucket_sort=True,
            pooling=args.pooling,
        )
    else:
        X_by_layer = layer_embeddings_esm(
            client=client,
            seqs=seqs,
            layer_ids=layer_ids,
            device=device,
            max_len=args.max_len,
        )

    
    for lid, X in X_by_layer.items():
        np.save(
            os.path.join(args.output, f"layer_{lid}.npy"),
            X.astype(np.float32, copy=False),
        )

    print(f"[done] outputs saved to {args.output}")
    print("  - sample_metadata_raw.csv / sample_metadata_filtered.csv")
    print("  - class_counts_before.json / class_counts_after.json")
    print("  - run_config.json")
    for lid in sorted(X_by_layer.keys()):
        print(f"  - layer_{lid}.npy  shape={X_by_layer[lid].shape}")


if __name__ == "__main__":
    main()
