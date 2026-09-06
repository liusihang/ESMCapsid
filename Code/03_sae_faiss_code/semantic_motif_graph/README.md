# Semantic-motif graph discovery

These scripts are the recovered source chain used to construct the frozen 364-community semantic-motif reference system:

1. `train_ivfpq_index.py`: train the padded 4,608-dimensional cosine-similarity IVFPQ index.
2. `search_ivfpq_knn.py`: add all reference token vectors and search the 50-neighbour graph.
3. `run_leiden_gpu.py`: construct the weighted undirected cuGraph graph and run Leiden clustering.

The manuscript-authoritative commands are:

```bash
python train_ivfpq_index.py \
  --input /path/to/sae_feats_sampled_109220seqs_csr.npz \
  --output_index /path/to/ivfpq_trained.index \
  --metric cosine --nlist 32768 --m 64 --nbits 8 \
  --train_size 2000000 --block_size 1000000 --seed 42

python search_ivfpq_knn.py \
  --input /path/to/sae_feats_sampled_109220seqs_csr.npz \
  --trained_index /path/to/ivfpq_trained.index \
  --output /path/to/knn_results_k50.npz \
  --metric cosine --k 50 --nprobe 64 --gpu 0 \
  --add_batch_size 200000 --query_batch_size 100000 --use_float16

python run_leiden_gpu.py \
  --knn /path/to/knn_results_k50.npz \
  --output /path/to/leiden_clusters.csv \
  --resolution 0.1 --min_weight 0.0 --gpu_id 0
```

The recovered raw HPC-source SHA-256 values are:

- `train_ivfpq_cpu.py`: `b30b0b3593db0924501e8690b1ea248f497b5490cdcfaa1c9e794cd68349c3b3`
- `search_ivfpq_gpu.py`: `2b4db3c45a785b5b4e6cc7cc37758844c67517191ec3d0e274a4166e56f5b56c`
- `run_leiden_gpu_acc.py`: `25b745bea087969b409ad7ed24486e90eab591713a7d1c48ffd805af5cc44e5d`

The public copies differ only by filename normalization and removal or translation of development-era comments and messages. The algorithm and defaults are unchanged.
