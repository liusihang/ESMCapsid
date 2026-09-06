# Figure 4B subgroup data

These tables are the submission-facing source data for the four Figure 4B dendrograms.
The cluster universe is defined directly by the current Figure 3 `combined_fold` classification.

- HK97-like: 67 embedding-defined clusters
- picorna-like: 11 embedding-defined clusters
- NCLDV-like: 7 embedding-defined clusters
- micro-like: 3 embedding-defined clusters (clusters 4, 10 and 19)
- Total displayed clusters: 88

Method:

- Each protein contributes its complete ordered semantic-motif string.
- Unigram, bigram and trigram counts are aggregated within sequence boundaries to the embedding-defined cluster level.
- Cluster profiles are TF-IDF weighted and L2 normalized.
- Pairwise profile similarity is cosine similarity; dendrogram distance is 1 - cosine similarity.
- Hierarchical clustering uses average linkage.
- Fold-internal subgroup number is selected by the maximum silhouette score over k=2..6, limited by fold size.

Public reconstruction uses the included per-fold cosine matrices and subgroup-assignment tables.
`panel_b_fold_map.csv` is the master Figure 3-to-Figure 4B cluster map,
`panel_b_subgroup_classification.csv` is the combined subgroup table, and
`panel_b_k_selection_silhouette.csv` contains all four silhouette scans.

Formal Figure 4 artwork is versioned separately from this data promotion.
