# Figure data and panel plotting scripts

This directory contains the plotting inputs and panel-level scripts for Figures 2–6 and Supplementary Figures S1–S6. All paths below are relative to `09_figure_data/`. Figure 1 contains final artwork only and has no plotting script in this package.

## Figure 2

| Panel | Script | Plotted content | Input data |
|---|---|---|---|
| 2A | `Figure2/scripts/build_panel_metrics.py --panel A` | Capsid-classification performance for the random split, 30% low-identity set, and family-held-out set | `Figure2/data/panel_a_random_split.csv`; `Figure2/data/panel_a_low_identity.csv`; `Figure2/data/panel_a_taxonomy_holdout.csv` |
| 2B | `Figure2/scripts/build_panel_b_embedding.py` | Two-dimensional embedding of capsid, other viral, structural, and cellular proteins | `Figure2/data/panel_b_embedding_coords.csv` |
| 2C | `Figure2/scripts/build_panel_metrics.py --panel C` | Masked-language-model accuracy and perplexity | `Figure2/data/panel_c_mlm_metrics.csv` |
| 2D | `Figure2/scripts/build_panel_metrics.py --panel D` | Downstream Macro-F1 before and after fine-tuning | `Figure2/data/panel_d_downstream_metrics.csv` |

## Figure 3

| Panel | Script | Plotted content | Input data |
|---|---|---|---|
| 3A | `Figure3/scripts/build_panel_a_repertoire.mjs` | Capsid sequence repertoire, fold expansion, HMM recovery, and predicted AAI50 clusters | `Figure3/data/panel_a_repertoire.csv` |
| 3B | `Figure3/scripts/build_panel_b_taxonomy.py` | Radial taxonomy tree colored by capsid architecture | `Figure3/data/panel_b_taxonomy/taxonomy_tree_nodes.csv`; `Figure3/data/panel_b_taxonomy/taxonomy_tree_edges.csv`; `Figure3/data/panel_b_taxonomy/family_top1_cluster.csv`; `Figure3/data/cluster_fold_classification.csv` |
| 3C | `Figure3/scripts/build_panel_c_similarity.py` | TM-score distributions for within-cluster, within-architecture, and between-architecture comparisons | `Figure3/data/panel_c_similarity/cluster_pair_summary.csv`; `Figure3/data/panel_c_similarity/category_significance.csv` |
| 3D | `Figure3/scripts/build_panel_d_structures.mjs` | Structure renderings for clusters 6 and 88 and their reference structures | `Figure3/data/panel_d_structures/cluster6_sim.pdb`; `Figure3/data/panel_d_structures/cluster88_sim.pdb`; `Figure3/data/panel_d_structures/c6.cif`; `Figure3/data/panel_d_structures/c88.cif`; `Figure3/scripts/render_structure.html` |
| 3D helper | `Figure3/scripts/render_structure.html` | NGL browser rendering page used by the Figure 3D script | One structure file URL supplied by `build_panel_d_structures.mjs` through the `file` query parameter |

## Figure 4

| Panel | Script | Plotted content | Input data |
|---|---|---|---|
| 4A | `Figure4/scripts/build_panel.py --panel A` | Motif-token positional distribution | `Figure4/data/panel_a_token_distribution.csv` |
| 4B | `Figure4/scripts/build_panel_b_subgroups.py` | Subgroup dendrograms for HK97-like, picorna-like, NCLDV-like, and micro-like clusters | `Figure4/data/panel_b_subgroups/{hk97,picorna,ncldv,micro}_cosine_similarity_matrix.csv`; `Figure4/data/panel_b_subgroups/{hk97,picorna,ncldv,micro}_subgroup_assignment.csv` |
| 4C | `Figure4/scripts/build_panel.py --panel C` | HK97-like and picorna-like motif position profiles | `Figure4/data/panel_c_hk97_position_profile.csv`; `Figure4/data/panel_c_picorna_position_profile.csv` |
| 4E | `Figure4/scripts/build_panel.py --panel E` | Structural scaffold metrics | `Figure4/data/panel_e_scaffold_metrics.csv` |

## Figure 5

| Panel | Script | Plotted content | Input data |
|---|---|---|---|
| 5A | `Figure5/scripts/build_panel_a_map.py` | Geographic distribution and ecosystem composition of sequences | `Seqs2Coordinates.csv` and `Seqs2Ecosystem.csv`, deposited on Figshare at [10.6084/m9.figshare.33440569](https://doi.org/10.6084/m9.figshare.33440569) and expected at `Figure5/data/source_data/` |
| 5B | `Figure5/scripts/build_panel_b_intersection.py` | Sequence and cluster sharing across ecosystem categories | `Figure5/data/panel_b_cluster_sharing.csv`; `Figure5/data/panel_b_hmm_annotation.csv` |

## Figure 6

| Panel | Script | Plotted content | Input data |
|---|---|---|---|
| 6A | `Figure6/scripts/build_panel.R --panel A` | Fold-group size, HMM recovery, and redundancy summary | `Figure6/data/panel_a_fold_summary.csv` |
| 6B | `Figure6/scripts/build_panel.R --panel B` | Ecosystem enrichment by fold group | `Figure6/data/panel_b_ecosystem_enrichment.csv`; `Figure6/data/panel_a_fold_summary.csv` for fold ordering |
| 6C | `Figure6/scripts/build_panel.R --panel C` | Fold-group, phylum, and genome-type associations with sub-ecosystems | `Figure6/data/panel_c_fold_by_sub_ecosystem_counts.csv`; `Figure6/data/panel_c_taxonomy_associations.csv`; `Figure6/data/panel_a_fold_summary.csv` for fold ordering |

## Supplementary figures

| Figure or panel | Script | Plotted content | Input data |
|---|---|---|---|
| S1, one model panel per run | `FigureS1/scripts/build_panel.py --model <model>` | t-SNE embedding of capsid and other proteins for ESMC 600M, ESM3O, ESM2 650M, or Profluent E1 600M | `FigureS1/data/{esmc_600m,esm3o,esm2_650m,profluent_e1_600m}/coords.csv` |
| S2 | `FigureS2/scripts/build_panel.py` | Layer-wise probe performance and the final two-head operating point | `FigureS2/data/layer_probe_metrics.csv`; `FigureS2/data/two_head_metrics.csv` |
| S3, mean-persistence panel | `FigureS3/scripts/build_panel.py --metric mean_persistence` | Mean HDBSCAN cluster persistence across layers and parameter combinations | `FigureS3/data/hdbscan_scan_metrics.csv` |
| S3, noise-percentage panel | `FigureS3/scripts/build_panel.py --metric percent_noise` | HDBSCAN noise percentage across layers and parameter combinations | `FigureS3/data/hdbscan_scan_metrics.csv` |
| S4 | `FigureS4/scripts/build_panel.py` | Fold-group motif-family heatmap based on mean TF-IDF | `FigureS4/data/core_HK97_like.csv`; `FigureS4/data/core_Pico_like.csv`; `FigureS4/data/core_NCLDV-like.csv`; `FigureS4/data/core_BTV-like.csv`; `FigureS4/data/core_Micro-like.csv` |
| S5A | `FigureS5/scripts/build_panel.py --panel A` | UMAP embedding of clusters colored by capsid architecture | `FigureS5/data/panel_a_embedding_coordinates_input.csv` |
| S5B | `FigureS5/scripts/build_panel.py --panel B` | Mean pairwise Jensen–Shannon divergence of motif positions by fold group | `FigureS5/data/panel_b_position_divergence.csv` |
| S6, one ecosystem panel per run | `FigureS6/scripts/build_panel.py --ecosystem <ecosystem>` | Cluster-accumulation curve for one main ecosystem | `FigureS6/data/cluster_accumulation.csv` |
