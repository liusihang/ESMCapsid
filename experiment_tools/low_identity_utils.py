from __future__ import annotations


def parse_cluster_mapping_rows(rows):
    return {row["prot_id"]: row["cluster_id"] for row in rows}


def build_grouped_cluster_folds(rows, n_splits: int):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["cluster_id"], []).append(row["prot_id"])

    clusters = sorted(grouped)
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if len(clusters) < n_splits:
        raise ValueError("not enough clusters for requested splits")

    cluster_order = sorted(
        clusters, key=lambda cluster: (-len(grouped[cluster]), cluster)
    )
    fold_clusters = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    for cluster in cluster_order:
        target = min(range(n_splits), key=lambda idx: (fold_sizes[idx], idx))
        fold_clusters[target].append(cluster)
        fold_sizes[target] += len(grouped[cluster])

    folds = []
    for i in range(n_splits):
        heldout_clusters = sorted(fold_clusters[i])
        test_positive_ids = sorted(
            prot_id for cluster in heldout_clusters for prot_id in grouped[cluster]
        )
        train_positive_ids = sorted(
            prot_id
            for cluster, prot_ids in grouped.items()
            if cluster not in heldout_clusters
            for prot_id in prot_ids
        )
        folds.append(
            {
                "fold": i + 1,
                "heldout_clusters": heldout_clusters,
                "train_positive_ids": train_positive_ids,
                "test_positive_ids": test_positive_ids,
            }
        )
    return folds
