from __future__ import annotations

from collections import Counter


LEVEL_SUFFIXES = {
    "Realm": "viria",
    "Kingdom": "virae",
    "Phylum": "viricota",
    "Class": "viricetes",
    "Family": "viridae",
}


def clean_taxonomy_value(level: str, value):
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    suffix = LEVEL_SUFFIXES[level]
    if not cleaned.endswith(suffix):
        return None
    return cleaned


def filter_valid_taxonomy_rows(rows, level: str):
    filtered = []
    for row in rows:
        cleaned = clean_taxonomy_value(level, row.get(level))
        if cleaned is None:
            continue
        new_row = dict(row)
        new_row[level] = cleaned
        filtered.append(new_row)
    return filtered


def select_overlap_capsid_ids(base_capsid_ids, taxonomy_rows):
    taxonomy_ids = {row["prot_id"] for row in taxonomy_rows if row.get("prot_id")}
    return set(base_capsid_ids) & taxonomy_ids


def filter_groups_by_min_size(rows, level: str, min_size: int, allowed_ids=None):
    allowed_ids = None if allowed_ids is None else set(allowed_ids)
    kept_rows = []
    for row in rows:
        if allowed_ids is not None and row.get("prot_id") not in allowed_ids:
            continue
        cleaned = clean_taxonomy_value(level, row.get(level))
        if cleaned is None:
            continue
        new_row = dict(row)
        new_row[level] = cleaned
        kept_rows.append(new_row)

    counts = Counter(row[level] for row in kept_rows)
    return [row for row in kept_rows if counts[row[level]] >= min_size]


def build_leave_one_taxon_out_splits(rows, level: str):
    cleaned_rows = filter_valid_taxonomy_rows(rows, level=level)
    grouped = {}
    for row in cleaned_rows:
        grouped.setdefault(row[level], []).append(row["prot_id"])

    splits = []
    for taxon in sorted(grouped):
        test_positive_ids = sorted(grouped[taxon])
        train_positive_ids = sorted(
            prot_id
            for other_taxon, prot_ids in grouped.items()
            if other_taxon != taxon
            for prot_id in prot_ids
        )
        splits.append(
            {
                "level": level,
                "heldout_taxon": taxon,
                "train_positive_ids": train_positive_ids,
                "test_positive_ids": test_positive_ids,
            }
        )
    return splits
