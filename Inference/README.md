# ESMCapsid Inference

**An ESM-only command-line toolkit for capsid candidate screening, sequence embeddings, and property prediction.**

ESMCapsid Inference combines the released ESMCapsid-S and ESMCapsid-C models with their existing prediction heads in a standalone inference workflow. It accepts protein sequences and produces per-sequence predictions, a candidate FASTA file, and sequence embeddings.

This toolkit is maintained separately from the [ESMCapsid research code](https://github.com/liusihang/ESMCapsid). Current version: **0.1.0**.

```text
Protein sequences → ESMCapsid-S screening → Candidates → ESMCapsid-C → Optional property heads
```

## Contents

- [Installation](#installation)
- [Models and property heads](#models-and-property-heads)
- [Quick start](#quick-start)
- [Input formats](#input-formats)
- [Output files](#output-files)
- [Command-line options](#command-line-options)
- [Development and testing](#development-and-testing)
- [Citation](#citation)
- [License](#license)

## Installation

Python **3.10 or later** is required. Package dependencies are installed automatically; version constraints are defined in [pyproject.toml](pyproject.toml).

From a local copy of the source:

```bash
cd /path/to/ESMCapsid/Inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .

esmcapsid --version
esmcapsid --help
```

For CUDA, first install a compatible PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/), then install this toolkit. The toolkit does not install GPU drivers or system CUDA libraries.

Installation is currently from source. No PyPI, Bioconda, or container release is provided.

## Models and property heads

### Local models and offline use

The toolkit uses the following released models:

| Model | Role | Release |
|---|---|---|
| ESMCapsid-S | Capsid candidate screening with the normal and hard-negative heads | [Shuofang127/ESMCapsid-S](https://huggingface.co/Shuofang127/ESMCapsid-S) |
| ESMCapsid-C | 1,152-dimensional sequence embeddings and representations for the existing property heads | [Shuofang127/ESMCapsid-C](https://huggingface.co/Shuofang127/ESMCapsid-C) |

Organize existing model directories, or symbolic links to them, as follows:

```text
models/
├── ESMCapsid-S/
│   ├── config.json
│   ├── model.safetensors
│   ├── modeling_esm_plusplus.py
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── heads/two_stage_layer16_hardneg/
│       ├── two_stage_config.json
│       ├── normal_head/
│       └── hardneg_head/
└── ESMCapsid-C/
    ├── config.json
    ├── model.safetensors
    ├── modeling_esm_plusplus.py
    ├── tokenizer.json
    └── tokenizer_config.json
```

Pass the **parent directory** to `--models`, not an individual model directory. Preserve the complete release files, including the classifier checkpoints and normalization arrays; copying only the encoder weights is insufficient.

All quick-start commands below use `--models models --offline`: they read existing files and do not download models. Replace `models` with your model parent directory.

- Screening alone requires S and its two screening heads.
- `embed` and `annotate` require C. `annotate` also requires property heads.
- For automatic online retrieval, omit both `--models` and `--offline`. The toolkit retrieves the release snapshots pinned for this version into the standard Hugging Face cache.
- With `--offline` but without `--models`, only the previously cached, pinned snapshots are used. Release identifiers are defined in [releases.py](src/esmcapsid/releases.py).

The models contain custom Transformers code and are loaded with `trust_remote_code=True`. Use only trusted release packages or local copies.

### Optional property heads

**The public ESMCapsid-C repository does not include property heads.** Property prediction requires a separately supplied head directory:

```text
property_heads/
├── realm/
│   ├── config.json
│   └── best_model.pth
├── family/
│   ├── config.json
│   └── best_model.pth
├── fold_label/
│   ├── config.json
│   └── best_model.pth
└── ...
```

The validated head package contains nine tasks:

- Taxonomic labels: `realm`, `kingdom`, `phylum`, `class`, `order`, and `family`.
- Other properties: `genome_label`, `host_group`, and `fold_label`.

Each head uses its existing class mapping and prediction threshold. The toolkit computes the matching input representation internally.

## Quick start

Run these commands from the project directory after preparing the models. Each output directory must be new.

### Screen a mixed protein collection

```bash
esmcapsid predict \
  --input proteins.faa \
  --models models --offline \
  --out screening \
  --screen-only --device cpu
```

This produces screening scores and decisions for the input records, plus a candidate FASTA file. C is not loaded.

### Screen and embed candidates

```bash
esmcapsid predict \
  --input proteins.faa \
  --models models --offline \
  --out candidate_embeddings \
  --device cpu
```

S screens all valid inputs; C processes only the candidates. C embeddings are saved by default. Add `--save-embeddings` to save S embeddings as well.

### Screen and predict candidate properties

```bash
esmcapsid predict \
  --input proteins.faa \
  --models models --offline \
  --property-heads /path/to/property_heads \
  --out candidate_properties \
  --device cpu
```

This combines screening, C encoding, and property prediction in one workflow. Screening-negative records remain in the results table but are not processed by C or the property heads.

### Annotate known or candidate capsid proteins

```bash
esmcapsid annotate \
  --input capsids.faa \
  --models models --offline \
  --property-heads /path/to/property_heads \
  --out annotations \
  --device cpu
```

This skips S and runs C with the property heads. `capsid_pred` is left empty because these inputs were not screened in this run.

For C embeddings without screening or property prediction:

```bash
esmcapsid embed \
  --input proteins.faa \
  --models models --offline \
  --out embeddings \
  --device cpu
```

The supplied [examples/proteins.faa](examples/proteins.faa) can replace the input in these commands for a small execution check. It contains toy sequences from the model cards, **not validated capsid-positive controls**. Zero screening candidates is a valid outcome.

## Input formats

Supported inputs are protein **FASTA, FASTA.gz, CSV, and CSV.gz**. Ground-truth labels are not required.

FASTA example:

```fasta
>protein_1 description
MSTNPKPQRKTKRNT
>protein_2
MKTIIALSYIFCLVFA
```

CSV defaults to the `prot_id` and `seq` columns:

```csv
prot_id,seq
protein_1,MSTNPKPQRKTKRNT
protein_2,MKTIIALSYIFCLVFA
```

Specify other column names explicitly:

```bash
esmcapsid predict \
  --input proteins.csv --id-column protein_id --sequence-column sequence \
  --models models --offline --screen-only \
  --out csv_screening
```

Input handling:

- Original IDs, complete FASTA headers, and record order are retained. Unique internal IDs prevent duplicate original IDs from overwriting results.
- Whitespace is removed and sequences are converted to uppercase. One terminal `*` is removed and recorded.
- Protein letters and common ambiguous or non-standard amino-acid letters are accepted. Empty sequences, gaps, internal `*`, and unsupported characters are marked invalid rather than silently replaced with `X`.
- Invalid records are excluded from inference but remain in a completed results table.



## Output files

All files are written under `--out`.

| File | Contents and conditions |
|---|---|
| `predictions.tsv` | One row per original input, with identities, lengths, processing states, screening scores, and optional property predictions |
| `capsid_candidates.faa` | Screening-positive records from `predict`; full-length cleaned sequences, not truncated fragments |
| `c_embeddings.npy` | Float32 C embeddings with shape `(N, 1152)`, when the C workflow is requested |
| `c_embedding_ids.tsv` | Embedding row indices mapped to internal IDs, original IDs, and input record indices |
| `s_embeddings.npy`, `s_embedding_ids.tsv` | Optional S embeddings for valid screening inputs, generated by `predict --save-embeddings` |
| `run.json` | Software version, key arguments, counts, warnings, and completion or failure state |
| `run.log` | Stage messages and runtime errors |

Candidate FASTA headers use unique internal IDs; the original identities remain in `predictions.tsv`. Embedding files contain only the records processed by that stage. Join embeddings through the corresponding `*_embedding_ids.tsv`, not through the original input row positions.

### Important fields and states

| Field | Meaning |
|---|---|
| `internal_id`, `original_id`, `original_header` | Unique internal identity and original input identity |
| `input_status`, `input_issue` | Input validity and reason for exclusion |
| `normal_score`, `hardneg_score` | S screening-head scores |
| `capsid_pred` | `1`: candidate; `0`: screened negative; empty: not screened |
| `screen_processed_aa`, `c_processed_aa` | Residues actually processed by each encoder |
| `screen_truncated`, `c_truncated` | `1` indicates truncation at that stage |
| `<task>_pred`, `<task>_score` | Property labels and model scores; present only when property heads are supplied |
| `property_status` | Distinguishes completion, no request, screening-negative inputs, absent heads, and invalid inputs |

**Check that `run.json` has `status=completed` before treating a run as complete.** Failed runs can leave partial tables or arrays. Early argument or input errors may occur before an output directory is created; those errors are reported in the terminal.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--input` | Required | Protein FASTA or CSV, optionally gzip-compressed |
| `--out` | Required | New output directory; existing directories are refused |
| `--models` | Unset | Local model parent; otherwise use pinned HF snapshots |
| `--offline` | Off | Read existing assets only; do not download from HF |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`; auto selects CUDA when available, otherwise CPU |
| `--batch-size` | `8` | Encoder batch size |
| `--property-heads` | Unset | Optional for `predict`; required for `annotate` |
| `--screen-only` | Off | `predict` only; cannot be combined with `--property-heads` |
| `--save-embeddings` | Off | `predict` only; also save S embeddings |
| `--id-column`, `--sequence-column` | `prot_id`, `seq` | CSV column names |
| `--screen-max-tokens` | `1022` | S tokenizer limit; `predict` only |
| `--c-max-tokens` | `786` | C tokenizer limit |

An explicit CUDA request fails if CUDA is unavailable; the toolkit does not silently switch devices. Changing token limits changes the processed sequence range and is not a new validation of model performance.

For complete command help:

```bash
esmcapsid predict --help
esmcapsid annotate --help
esmcapsid embed --help
```
## Development and testing

Install development dependencies from the source directory:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Code tests use small or synthetic inputs and do not download full models. [scripts/validate_public_release.py](scripts/validate_public_release.py) performs real-weight comparisons using existing models and reference functions from the original code.

## Citation

For work using ESMCapsid models, cite the study listed in the model cards:

Liu, S., Xia, S., and Wang, H. *Capsid-specialized protein language models reveal higher-order viral architecture from sequence*. bioRxiv (2026). DOI: [10.64898/2026.09.06.749605](https://doi.org/10.64898/2026.09.06.749605).

For base-model and implementation references, consult the [ESMCapsid-S model card](https://huggingface.co/Shuofang127/ESMCapsid-S) and [ESMCapsid-C model card](https://huggingface.co/Shuofang127/ESMCapsid-C).

This inference toolkit does not currently have a separate software DOI. Record its version separately from the version of the research-code repository.

## License

Model weights are governed by the `LICENSE` and `NOTICE.txt` in their respective repositories. This toolkit does not change or relicense those weights.

A separate code `LICENSE` has not yet been specified for this toolkit. Do not assume that the model licenses also license this toolkit, or that the research code, base models, and this toolkit share the same license.
