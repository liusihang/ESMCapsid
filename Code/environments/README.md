# HPC environment package inventories

These files record the installed package name, version, build string and channel observed in the three HPC environments used by the ESMCapsid workflows. They are package inventories, not environment installation scripts.

| Inventory | HPC environment | Packages |
|---|---|---:|
| `esm_packages.tsv` | ESM model training and inference | 218 |
| `faiss_packages.tsv` | FAISS and RAPIDS analysis | 367 |
| `vicapsid_packages.tsv` | Integrated ESMCapsid workflow | 304 |

Inventory filenames and runtime environment identifiers are retained to match the original environment records.

Captured from the corresponding HPC Conda environments on 2026-08-27 using `conda list --json`. GPU drivers, Slurm modules and host-level CUDA libraries are not Conda packages and are therefore outside these inventories.
