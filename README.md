# ESMCapsid

This repository contains the ESMCapsid code, inference toolkit, and figure data supporting the ESMCapsid study.

| Directory | Contents |
|---|---|
| [`Inference/`](Inference/) | Installable ESM-only toolkit for capsid screening, sequence embeddings, and property prediction. |
| [`Code/`](Code/) | Training, evaluation, SAE/FAISS, structure-analysis, and inference-pipeline code. |
| [`Figure_data/`](Figure_data/) | Panel-level plotting scripts, plotting inputs, and final figure files. |

## Released models

- [ESMCapsid-S](https://huggingface.co/Shuofang127/ESMCapsid-S): screening encoder and its two screening heads.
- [ESMCapsid-C](https://huggingface.co/Shuofang127/ESMCapsid-C): representation encoder and
  [nine pretrained property heads](https://huggingface.co/Shuofang127/ESMCapsid-C/tree/main/heads/property_heads)
  for taxonomy, genome labels, host groups, and fold labels.

See the [Inference README](Inference/README.md#models-and-property-heads) for
installation, a heads-only download, and offline use with existing models.

## External Figure 5 data

The two large source tables used by Figure 5A are deposited on Figshare at [10.6084/m9.figshare.33440569](https://doi.org/10.6084/m9.figshare.33440569):

- `Seqs2Coordinates.csv`
- `Seqs2Ecosystem.csv`

After downloading, place both files in `Figure_data/Figure5/data/source_data/`.

## License

The original ESMCapsid research code in `Code/`, the toolkit in `Inference/`,
and the plotting scripts in `Figure_data/` are released under the
[MIT License](LICENSE). Third-party code and dependencies retain their own
licenses. This code license does not assign a license to scientific data,
figure artwork, or model weights.

The ESMCapsid-S and ESMCapsid-C weights and their prediction heads are
distributed separately on Hugging Face. They are subject to the terms in
their respective model repositories, including `LICENSE` and `NOTICE.txt`:
[ESMCapsid-S](https://huggingface.co/Shuofang127/ESMCapsid-S/tree/main) and
[ESMCapsid-C](https://huggingface.co/Shuofang127/ESMCapsid-C/tree/main).
The model notices include the EvolutionaryScale Cambrian Non-Commercial
License Agreement. MIT licensing of this code does not grant permission for
commercial use of those models or remove their upstream restrictions.
