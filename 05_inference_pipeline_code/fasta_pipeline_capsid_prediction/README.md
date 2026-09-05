# FastaPipeline

Canonical FASTA → PLM → SAE → semantic motif / sequence kNN pipeline.

## Layout

- `pipeline/`: canonical Python parts and shared utilities.
- `config/reference_bundle.json`: default HPC reference bundle.
- `backends/`: PLM, SAE, semantic-motif, pooling, and kNN backends.
- `runs/<run_id>/`: runtime outputs, created when the pipeline runs.

## Parts

- `prepare`: parse FASTA, clean sequences, deduplicate IDs, chunk input.
- `embed`: extract ESMCapsid-C layer-28 residue representations.
- `sae`: run token-level SAE inference with the pretrained SAE model.
- `map_semantic_motif`: map token SAE features to the packaged 364-community Leiden reference system.
- `pool_seq`: mean-pool token SAE features into sequence features.
- `predict_seq`: run sequence-level FAISS kNN label transfer.
- `report`: merge chunk results into run-level final reports.

## Quick start

```bash
python3 /absolute/path/to/FastaPipeline/run_fasta_pipeline.py \
  --fasta /path/to/input.fasta \
  --run-dir /absolute/path/to/FastaPipeline/runs/demo_run \
  --config /absolute/path/to/FastaPipeline/config/reference_bundle.json \
  --parts all
```

Run only GPU parts:

```bash
python3 /absolute/path/to/FastaPipeline/run_fasta_pipeline.py \
  --run-dir /absolute/path/to/FastaPipeline/runs/demo_run \
  --config /absolute/path/to/FastaPipeline/config/reference_bundle.json \
  --parts embed,sae,map_semantic_motif
```

Run only CPU parts:

```bash
python3 /absolute/path/to/FastaPipeline/run_fasta_pipeline.py \
  --run-dir /absolute/path/to/FastaPipeline/runs/demo_run \
  --config /absolute/path/to/FastaPipeline/config/reference_bundle.json \
  --parts pool_seq,predict_seq,report
```

## Canonical outputs

- `runs/<run_id>/chunks/chunk_XXXXX/00_input/input.normalized.fasta`
- `runs/<run_id>/chunks/chunk_XXXXX/10_embed/sequence_manifest.filtered.csv`
- `runs/<run_id>/chunks/chunk_XXXXX/20_sae/sae_token_features.npz`
- `runs/<run_id>/chunks/chunk_XXXXX/30_semantic_motif/token_semantic_motif_table.csv`
- `runs/<run_id>/chunks/chunk_XXXXX/40_seq/sequence_sae_mean_features.npz`
- `runs/<run_id>/chunks/chunk_XXXXX/50_seq_pred/sequence_knn_predictions.csv`
- `runs/<run_id>/60_report/final_sequence_predictions.csv`
- `runs/<run_id>/60_report/final_token_semantic_motif.parquet`

## Preflight

Run the path check before a real model job:

```bash
python3 preflight.py \
  --config /absolute/path/to/reference_bundle.json \
  --parts embed,sae,map_semantic_motif,pool_seq,predict_seq
```

The check resolves archive-relative backend scripts and fails early on unresolved
model, index, Python-environment, or reference paths.
