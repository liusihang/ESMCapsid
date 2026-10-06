# ESMCapsid

This repository contains the ESMCapsid code, inference toolkit, and figure data supporting the ESMCapsid study.

| Directory | Contents |
|---|---|
| [`Inference/`](Inference/) | Installable ESM-only toolkit for capsid screening, sequence embeddings, and property prediction. |
| [`Code/`](Code/) | Training, evaluation, SAE/FAISS, structure-analysis, and inference-pipeline code. |
| [`Figure_data/`](Figure_data/) | Panel-level plotting scripts, plotting inputs, and final figure files. |

## External Figure 5 data

The two large source tables used by Figure 5A are deposited on Figshare at [10.6084/m9.figshare.33440569](https://doi.org/10.6084/m9.figshare.33440569):

- `Seqs2Coordinates.csv`
- `Seqs2Ecosystem.csv`

After downloading, place both files in `Figure_data/Figure5/data/source_data/`.
