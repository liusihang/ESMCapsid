# ESMCapsid code archive

This archive contains the Python, Shell and Slurm source code for the ESMCapsid workflow. Runtime models, training data, benchmark tables, structure files and reference databases are supplied separately through the paths listed in `05_inference_pipeline_code/fasta_pipeline_capsid_prediction/config/reference_bundle.json` and the input arguments of each script.

The original research code is released under the [MIT License](../LICENSE).
Third-party code and dependencies retain their own licenses. Model weights
and prediction heads are separate assets governed by the `LICENSE` and
`NOTICE.txt` in their Hugging Face repositories, not by this code license.

## Directory map

| Directory | Responsibility |
|---|---|
| `01_training_code/` | ESMCapsid-C, ESMCapsid-S, classifier-head and property-head training entry points. |
| `02_evaluation_code/` | Representation benchmarks, low-identity evaluation, taxonomy-held-out evaluation and layer scans. |
| `03_sae_faiss_code/` | SAE sample preparation/training/inference and semantic-motif FAISS/Leiden construction. |
| `04_structure_analysis_code/` | TM-align coherence, motif-to-shell mapping and structural evidence summaries. |
| `05_inference_pipeline_code/` | Stage-1 screening, Stage-2 representation extraction, classifier prediction and the FASTA inference pipeline. |
| `experiment_tools/` | Shared training, split, threshold and linear-probe helpers. |
| `scripts/` | Shared structure-analysis utilities and repository-relative path helpers. |
| `environments/` | Package inventories for the ESM, FAISS/RAPIDS and integrated ESMCapsid workflow environments. |

## `01_training_code/`

| File | Responsibility |
|---|---|
| `ds_config.json` | DeepSpeed ZeRO Stage-2 configuration for ESMCapsid-C training. |
| `train_esmcapsid_c.py` | Continued masked-language-model training for ESMCapsid-C. |
| `train_esmcapsid_s_lora_mean_supcon.py` | LoRA encoder training with masked-mean pooling and supervised-contrastive learning for ESMCapsid-S. |
| `train_dual_stage_low_fpr.py` | Training of the Layer-16 normal and hard-negative classifier heads. |
| `train_capsid_property_heads.py` | Training entry point for taxonomy, architecture, genome-type and host-category heads. |
| `train_property_heads_layer33.py` | Training of Layer-33 sequence-property heads. |
| `prepare_dual_layer_low_fpr_jobs.py` | Generation of job directories and scheduler scripts for dual-stage head training and evaluation. |

## `02_evaluation_code/`

| File | Responsibility |
|---|---|
| `prepare_embedding_linear_probe.py` | Preparation of frozen-representation benchmark embeddings. |
| `linear_probe_cv_gpu.py` | GPU linear-probe cross-validation and random-split evaluation. |
| `linear_probe_utils.py` | Shared linear-probe data loading, fitting and metric utilities. |
| `build_low_identity_clusters.py` | Construction of low-identity sequence clusters for held-out evaluation. |
| `prepare_low_identity_jobs.py` | Generation of low-identity evaluation jobs. |
| `run_low_identity_capsid.py` | Execution of low-identity held-out evaluation. |
| `summarize_low_identity_results.py` | Aggregation of low-identity evaluation outputs. |
| `prepare_taxonomy_heldout_jobs.py` | Generation of taxonomy-held-out evaluation jobs. |
| `run_taxonomy_heldout_capsid.py` | Execution of family-held-out evaluation. |
| `taxonomy_heldout_utils.py` | Shared family-held-out split and label utilities. |
| `summarize_taxonomy_heldout_results.py` | Aggregation of taxonomy-held-out outputs. |
| `run_binary_layer_scan.py` | Layer-wise binary probe scan. |
| `summarize_binary_layer_scan.py` | Aggregation of layer-scan outputs. |
| `summarize_capsid_binary_results.py` | Summary of binary capsid benchmark results. |
| `three_group_eval.py` | MLM-loss and pseudo-perplexity evaluation across model groups. |

## `03_sae_faiss_code/`

| File or directory | Responsibility |
|---|---|
| `sae_d4604_k64_preprocess_config.json` | Configuration for SAE training-sample preparation. |
| `preprocess_sae_training_samples.py` | Sampling and preprocessing of token embeddings for SAE training. |
| `build_hard_subset_150k.py` | Construction of the fixed 150,000-protein SAE training subset. |
| `build_hard_subset_150k.sbatch` | Slurm launcher for the hard-subset construction. |
| `run_hard_subset_preprocessing.sbatch` | Slurm launcher for SAE sample preprocessing. |
| `run_sae_training.sh` | Shell launcher for hybrid SAE training. |
| `train_sae_hybrid.py` | Hybrid SAE model training and checkpoint export. |
| `infer_sae_token_activations.py` | Token-level SAE activation inference. |
| `semantic_motif_graph/` | Formal FAISS IVFPQ search and Leiden graph-construction source. |

### `03_sae_faiss_code/semantic_motif_graph/`

| File | Responsibility |
|---|---|
| `train_ivfpq_index.py` | Training and serialization of the IVFPQ index. |
| `search_ivfpq_knn.py` | Reference-vector insertion and k-nearest-neighbour search. |
| `run_leiden_gpu.py` | Weighted graph construction and Leiden community assignment. |
| `README.md` | Command-line interface and required input/output file names for the semantic-motif graph stages. |

## `04_structure_analysis_code/`

| File | Responsibility |
|---|---|
| `requirements_structure.txt` | Python dependencies for the structure-analysis entry points. |
| `run_structure_coherence_tmalign.py` | TM-align structural-coherence and permutation analysis. |
| `assembly_motif_structure_analysis.py` | Motif-to-assembly mapping and shell-level structural analysis. |
| `map_group_core_motifs_to_assemblies.py` | Mapping group core motifs onto capsid assemblies. |
| `analyze_cluster_core_robust_evidence.py` | Robustness analysis for cluster-core structural evidence. |
| `summarize_cluster_core_quantitative_evidence.py` | Quantitative summary of cluster-core structural evidence. |

## `05_inference_pipeline_code/`

| File | Responsibility |
|---|---|
| `run_capsid_stage1_dual_stage.py` | End-to-end Stage-1 candidate screening with the two Layer-16 heads. |
| `run_capsid_stage2_esmcapsid_c.py` | Stage-2 ESMCapsid-C token extraction and sequence-level representation generation. |
| `extract_sequence_embeddings.py` | Generic sequence-level PLM embedding extraction. |
| `extract_capsid_stage2_embeddings.py` | Stage-2 embedding extraction wrapper. |
| `predict_capsid_stage1.py` | Stage-1 prediction command. |
| `predict_layer16_two_head.py` | Layer-16 normal and hard-negative two-head prediction. |
| `predict_capsid_dual_head.py` | Dual-head prediction wrapper. |
| `predict_dual_stage_classifier_packaged.py` | Prediction using the packaged dual-stage classifier. |
| `predict_dual_stage_classifier_benchmark.py` | Benchmark-workspace dual-stage classifier prediction. |
| `export_dual_stage_classifier.py` | Export of classifier heads and normalization metadata. |
| `predict_capsid_properties.py` | Sequence-property prediction. |
| `predict_property_heads_layer33.py` | Layer-33 property-head prediction. |
| `predict_capsid_semantic_motifs.py` | Semantic-motif prediction wrapper. |

### `05_inference_pipeline_code/fasta_pipeline_capsid_prediction/`

| File or directory | Responsibility |
|---|---|
| `run_fasta_pipeline.py` | Top-level orchestration for FASTA/CSV inference. |
| `preflight.py` | Command-line entry point for runtime asset and executable path checks. |
| `pipeline/` | Resumable pipeline parts and shared data contracts. |
| `backends/` | Model, SAE, FAISS, pooling and sequence-kNN backend adapters. |
| `config/reference_bundle.json` | Machine-local configuration template for executables and external model/reference paths. |
| `smoke_requirements.txt` | Dependencies for the local orchestration example. |
| `smoke_test.py` | Deterministic pipeline fixture using generated mock backends. |
| `README.md` | FASTA pipeline command-line reference. |

### `05_inference_pipeline_code/fasta_pipeline_capsid_prediction/pipeline/`

| File | Responsibility |
|---|---|
| `common.py` | Shared configuration, manifests, chunk metadata and pipeline utilities. |
| `part00_prepare_input.py` | FASTA/CSV normalization, sequence manifest creation and chunk preparation. |
| `part10_extract_plm_tokens.py` | Layer-28 token embedding stage. |
| `part20_infer_sae_tokens.py` | Token-level SAE activation stage. |
| `part30_map_semantic_motif.py` | FAISS semantic-motif label transfer and confidence generation. |
| `part40_pool_sequence_features.py` | Mean pooling of token features into sequence features. |
| `part50_predict_sequence_knn.py` | Sequence-level FAISS kNN label transfer. |
| `part60_build_report.py` | Merging of sequence and token reports into final CSV/Parquet outputs. |
| `preflight.py` | Library function for selected-part configuration checks. |

### `05_inference_pipeline_code/fasta_pipeline_capsid_prediction/backends/`

| File | Responsibility |
|---|---|
| `extract_plm_tokens.py` | ESMCapsid-C token extraction backend. |
| `infer_sae_tokens.py` | SAE activation backend adapter. |
| `map_semantic_motifs.py` | FAISS/Leiden semantic-motif mapping backend. |
| `pool_sequence_features.py` | Token-to-sequence pooling backend. |
| `predict_sequence_knn.py` | Sequence-level kNN prediction backend. |
| `__init__.py` | Backend package marker. |

## `experiment_tools/`

| File | Responsibility |
|---|---|
| `layer_selection_utils.py` | Layer-selection and threshold utilities. |
| `linear_probe_cv_gpu.py` | Shared GPU linear-probe implementation. |
| `linear_probe_utils.py` | Shared linear-probe fold and metric utilities. |
| `low_identity_utils.py` | Low-identity split utilities. |
| `taxonomy_heldout_utils.py` | Taxonomy-held-out split utilities. |
| `train_dual_stage_low_fpr.py` | Importable dual-stage head training implementation. |
| `vicapsid_training_schedule.py` | Step-based evaluation and checkpoint schedule. |
| `__init__.py` | Helper package marker. |

## `scripts/`

| File | Responsibility |
|---|---|
| `common/assignment_groups.py` | Structural-analysis group assignment loader. |
| `common/sae_paths.py` | Repository-relative SAE analysis paths. |
| `common/__init__.py` | Common utility package marker. |
| `__init__.py` | Scripts package marker. |

## External files required at runtime

The code package uses the following external files and directories. Set their locations in command-line arguments, scheduler variables or `config/reference_bundle.json`.

| External item | Required contents | Used by |
|---|---|---|
| `/path/to/input/input.fasta` or `/path/to/input/input.csv` | Input protein sequences; CSV input uses the column expected by the selected command. | FASTA pipeline and Stage-1/Stage-2 runners |
| `/path/to/models/ESMCapsid-S_model/` | ESMCapsid-S configuration, tokenizer, custom model code and model weights. | Stage-1 screening and Layer-16 embedding extraction |
| `/path/to/models/ESMCapsid-C_model/` | ESMCapsid-C configuration, tokenizer, custom model code and model weights. | Stage-2 and Layer-28/Layer-33 representation extraction |
| `/path/to/models/two_stage_layer16_hardneg/` | `two_stage_config.json`, `normal_head/` and `hardneg_head/` with checkpoints and normalization arrays. | Two-head Stage-1 prediction |
| `/path/to/models/sae_D4604_K64/sae_model.pt` | D4604/K64 SAE checkpoint. | SAE token inference |
| `/path/to/models/sae_D4604_K64/global_stats.npz` | SAE centering and rescaling statistics. | SAE token inference |
| `/path/to/reference/semantic_motif/sae_feats_sampled_109220seqs_csr.npz` | Reference SAE feature matrix. | IVFPQ/semantic-motif mapping |
| `/path/to/reference/semantic_motif/ivfpq_trained.index` | Trained FAISS IVFPQ index. | Semantic-motif mapping |
| `/path/to/reference/semantic_motif/leiden_clusters.npy` | Reference Leiden community labels. | Semantic-motif mapping |
| `/path/to/reference/sequence_knn/filtered_mean_seq_feats.npz` | Reference sequence-level feature matrix. | Sequence-level kNN prediction |
| `/path/to/reference/sequence_knn/filtered_mean_clusters.csv` | Reference sequence-level labels with the configured label column. | Sequence-level kNN prediction |
| `/path/to/training/token_backend/` | Token arrays, offsets, lengths and sequence metadata for SAE preparation/training. | `03_sae_faiss_code/` |
| `/path/to/training/manifests/` | Training, validation, benchmark and held-out split manifests. | `01_training_code/` and `02_evaluation_code/` |
| `/path/to/structures/` | Structure manifest, coordinate files and prediction/reference reports. | `04_structure_analysis_code/` |
| `/path/to/tools/TMalign` | TMalign executable or command available on `PATH`. | TM-align structure analysis |

## Configuration path placeholders

`05_inference_pipeline_code/fasta_pipeline_capsid_prediction/config/reference_bundle.json` is a path template. Replace every `/path/to/...` value with paths in the execution environment while preserving the code-relative backend script paths.

The main configurable path groups are:

```text
ESM executable          /path/to/envs/esm/bin/python
FAISS executable        /path/to/envs/faiss/bin/python
ESMCapsid-S model       /path/to/models/ESMCapsid-S_model
ESMCapsid-C model       /path/to/models/ESMCapsid-C_model
SAE checkpoint          /path/to/models/sae_D4604_K64/sae_model.pt
SAE statistics          /path/to/models/sae_D4604_K64/global_stats.npz
Semantic-motif assets   /path/to/reference/semantic_motif/
Sequence-kNN assets     /path/to/reference/sequence_knn/
```

`05_inference_pipeline_code/fasta_pipeline_capsid_prediction/README.md` contains the command-line argument reference for the FASTA pipeline. Each training, evaluation and structure-analysis script exposes its own input and output arguments or scheduler variables.
