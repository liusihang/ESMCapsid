#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Leiden clustering with int64-safe edge indices.
Implementation details:
1. Memory-aware NPZ loading.
2. GPU-side filtering with cuDF and CuPy.
3. Reduced intermediate copying.
4. Batched host-to-device transfer.
"""
import argparse
import numpy as np
import cudf
import cupy as cp
import cugraph
import rmm
import time
import gc
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Run Leiden clustering on GPU (Int64 Safe)")
    p.add_argument('--knn', type=str, required=True, help='Path to knn_results.npz')
    p.add_argument('--output', type=str, default='leiden_clusters.csv')
    p.add_argument('--resolution', type=float, default=1.0)
    p.add_argument('--min_weight', type=float, default=0.0)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--chunk_size', type=int, default=50_000_000, 
                   help='Chunk size for GPU transfer (edges per chunk)')
    return p.parse_args()


def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_id)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    pynvml.nvmlShutdown()
    
    
    initial_pool = int(gpu_mem * 0.8)
    
    try:
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=True,
            initial_pool_size=initial_pool,  
            devices=[args.gpu_id]
        )
        print(f" RMM Managed Memory enabled on GPU {args.gpu_id}")
        print(f"  Initial pool: {initial_pool / 1e9:.1f} GB")
    except Exception as e:
        print(f"Warning: RMM initialization failed ({e}), using default.")
        rmm.reinitialize(managed_memory=True, pool_allocator=True, devices=[args.gpu_id])

    
    cp.cuda.Device(args.gpu_id).use()
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print(f"Loading KNN data from {args.knn}...")
    t_load = time.time()
    
    
    data = np.load(args.knn, mmap_mode='r')
    
    n_samples = int(data['n_samples']) if 'n_samples' in data else data['neighbors'].shape[0]
    k = int(data['k']) if 'k' in data else data['neighbors'].shape[1]
    metric = str(data['metric']) if 'metric' in data else 'l2'
    
    total_edges_raw = n_samples * k
    print(f"  Nodes: {n_samples:,}, k: {k}")
    print(f"  Raw edges: {total_edges_raw:,}")
    print(f"  Load time: {time.time() - t_load:.2f}s")
    
    use_int64 = total_edges_raw > 2_000_000_000 or n_samples > 2_000_000_000
    dtype_idx = np.int64 if use_int64 else np.int32
    
    if use_int64:
        print("  ! Using INT64 indices (edge/node count exceeds int32 safe range)")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("Building edge list with GPU-side filtering...")
    t_build = time.time()
    
    
    edge_mem_bytes = total_edges_raw * (8 + 8 + 4)  # src(int64) + dst(int64) + weight(float32)
    available_gpu_mem = initial_pool * 0.6  
    
    need_chunking = edge_mem_bytes > available_gpu_mem
    
    if not need_chunking:
        
        print("  Mode: Full transfer (sufficient GPU memory)")
        
        
        neighbors_flat = data['neighbors'].astype(dtype_idx).ravel()
        distances_flat = data['distances'].astype(np.float32).ravel()
        
        
        sources = np.broadcast_to(
            np.arange(n_samples, dtype=dtype_idx)[:, np.newaxis], 
            (n_samples, k)
        ).ravel().copy()  
        
        
        src_gpu = cudf.Series(sources)
        dst_gpu = cudf.Series(neighbors_flat)
        weight_gpu = cudf.Series(distances_flat)
        
        del sources, neighbors_flat, distances_flat
        
        
        if metric == 'l2':
            print("  Converting L2 distances to similarities (GPU)...")
            weight_gpu = 1.0 / (1.0 + weight_gpu)
        
        
        mask = (dst_gpu >= 0) & (dst_gpu < n_samples)
        if args.min_weight > 0:
            mask = mask & (weight_gpu > args.min_weight)
        
        
        src_gpu = src_gpu[mask]
        dst_gpu = dst_gpu[mask]
        weight_gpu = weight_gpu[mask]
        
        del mask
        
        df = cudf.DataFrame({'src': src_gpu, 'dst': dst_gpu, 'weight': weight_gpu})
        del src_gpu, dst_gpu, weight_gpu
        
    else:
        
        print(f"  Mode: Chunked transfer (chunk_size={args.chunk_size:,})")
        
        chunk_size = args.chunk_size
        dfs = []
        
        for start_node in range(0, n_samples, chunk_size // k):
            end_node = min(start_node + chunk_size // k, n_samples)
            n_chunk = end_node - start_node
            
            
            neighbors_chunk = data['neighbors'][start_node:end_node].astype(dtype_idx).ravel()
            distances_chunk = data['distances'][start_node:end_node].astype(np.float32).ravel()
            
            
            sources_chunk = np.repeat(
                np.arange(start_node, end_node, dtype=dtype_idx), k
            )
            
            
            src_gpu = cudf.Series(sources_chunk)
            dst_gpu = cudf.Series(neighbors_chunk)
            weight_gpu = cudf.Series(distances_chunk)
            
            del sources_chunk, neighbors_chunk, distances_chunk
            
            
            if metric == 'l2':
                weight_gpu = 1.0 / (1.0 + weight_gpu)
            
            
            mask = (dst_gpu >= 0) & (dst_gpu < n_samples)
            if args.min_weight > 0:
                mask = mask & (weight_gpu > args.min_weight)
            
            chunk_df = cudf.DataFrame({
                'src': src_gpu[mask],
                'dst': dst_gpu[mask],
                'weight': weight_gpu[mask]
            })
            dfs.append(chunk_df)
            
            del src_gpu, dst_gpu, weight_gpu, mask
            
            if (end_node - start_node) % (chunk_size // k * 10) == 0:
                print(f"    Processed {end_node:,} / {n_samples:,} nodes")
        
        
        df = cudf.concat(dfs, ignore_index=True)
        del dfs
    
    
    del data
    gc.collect()
    
    num_edges = len(df)
    print(f"  Valid edges: {num_edges:,}")
    print(f"  Build time: {time.time() - t_build:.2f}s")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("Building cugraph...")
    t_graph = time.time()
    
    G = cugraph.Graph()
    
    try:
        G.from_cudf_edgelist(
            df,
            source='src',
            destination='dst',
            edge_attr='weight',
            renumber=False
        )
    except Exception as e:
        print(f"  renumber=False failed: {e}")
        print("  Retrying with renumber=True...")
        G.from_cudf_edgelist(
            df,
            source='src',
            destination='dst',
            edge_attr='weight',
            renumber=True
        )
    
    del df
    gc.collect()  
    
    print(f"  Graph build time: {time.time() - t_graph:.2f}s")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print(f"Running Leiden (resolution={args.resolution})...")
    t_leiden = time.time()
    
    try:
        parts, modularity = cugraph.leiden(G, resolution=args.resolution)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n!!! OOM during Leiden !!!")
            print("Suggestions:")
            print("  1. Reduce resolution parameter")
            print("  2. Increase system swap space")
            print("  3. Use a GPU with more VRAM")
        raise
    
    elapsed_leiden = time.time() - t_leiden
    print(f"  Leiden time: {elapsed_leiden:.2f}s")
    print(f"  Modularity: {modularity:.4f}")
    print(f"  Clusters found: {parts['partition'].nunique()}")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print(f"Saving results to {args.output}...")
    t_save = time.time()
    
    
    final_labels = cp.full(n_samples, -1, dtype=cp.int32)
    
    v_idx = cp.asarray(parts['vertex'].values)
    c_idx = cp.asarray(parts['partition'].values)
    
    
    max_vertex = int(v_idx.max())
    if max_vertex >= n_samples:
        print(f"  Warning: max vertex ID ({max_vertex}) >= n_samples ({n_samples})")
    
    
    final_labels[v_idx] = c_idx
    
    
    output_npy = args.output.replace('.csv', '.npy')
    np.save(output_npy, cp.asnumpy(final_labels))
    print(f"  Saved: {output_npy}")
    
    
    
    if args.output.endswith('.csv'):
        
        parts.to_csv(args.output, index=False)
        print(f"  Saved: {args.output}")
    
    print(f"  Save time: {time.time() - t_save:.2f}s")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total nodes:    {n_samples:,}")
    print(f"  Total edges:    {num_edges:,}")
    print(f"  Clusters:       {parts['partition'].nunique()}")
    print(f"  Modularity:     {modularity:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()