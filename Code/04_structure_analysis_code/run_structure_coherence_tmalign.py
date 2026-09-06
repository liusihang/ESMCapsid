#!/usr/bin/env python3
"""Run the AF3 TM-align structural-coherence analysis and significance plot."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/path/to/structure_validation")
MANIFEST_CSV = (
    WORKSPACE
    / "inputs/selected_structure_manifest.csv"
)
PRED_ROOT = Path("/path/to/structure_validation/predicted_structures")
OUTDIR = WORKSPACE / "outputs/tmalign_structure_coherence"
STAGE_DIR = WORKSPACE / "work/tmalign_structure_coherence_stage"
LOCAL_BEST_DIR = STAGE_DIR / "best_models"

REMOTE_HOST = "hpc.example.org"
REMOTE_ROOT = (
    "/path/to/vicapsid/FormalData/Part4FoldStats/Analysis/"
    "tmalign_structure_coherence"
)
REMOTE_INPUT_DIR = f"{REMOTE_ROOT}/inputs/best_models"
REMOTE_WORKDIR = f"{REMOTE_ROOT}/results"
TMALIGN = "TMalign"
THREADS = 16
REPLACE_STAGE = False
REPLACE_REMOTE = False

CATEGORY_LABELS = {
    "within_cluster": "Within cluster",
    "between_clusters_same_group": "Between clusters\nsame group",
    "between_groups": "Between groups",
}
PLOT_ORDER = [
    "within_cluster",
    "between_clusters_same_group",
    "between_groups",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the selected no-HMM AF3 TMalign comparison pipeline."
    )
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument(
        "--tmalign",
        default="TMalign",
        help="TMalign command name or absolute path on the remote host.",
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--replace-stage",
        action="store_true",
        help="Replace an existing local staging directory.",
    )
    parser.add_argument(
        "--replace-remote",
        action="store_true",
        help="Replace an existing remote analysis directory.",
    )
    return parser.parse_args()


def run(
    args: list[str],
    *,
    input_text: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
    )
    if completed.returncode:
        parts = [
            f"Command failed: {' '.join(args)}",
            f"exit code: {completed.returncode}",
        ]
        if completed.stdout:
            parts.append(f"stdout:\n{completed.stdout}")
        if completed.stderr:
            parts.append(f"stderr:\n{completed.stderr}")
        raise RuntimeError("\n".join(parts))
    return completed


def ssh_python(code: str) -> str:
    return run(["ssh", "-Tq", REMOTE_HOST, "python3", "-"], input_text=code).stdout


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    return {
        "n_pairs": int(arr.size),
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std(ddof=0)),
        "min_score": float(arr.min()),
        "q10_score": float(np.quantile(arr, 0.10)),
        "q25_score": float(np.quantile(arr, 0.25)),
        "q75_score": float(np.quantile(arr, 0.75)),
        "q90_score": float(np.quantile(arr, 0.90)),
        "max_score": float(arr.max()),
    }


def load_best_prediction_catalog() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not MANIFEST_CSV.exists():
        raise RuntimeError(f"Manifest missing: {MANIFEST_CSV}")
    if not PRED_ROOT.exists():
        raise RuntimeError(f"Prediction root missing: {PRED_ROOT}")

    manifest_rows = read_csv(MANIFEST_CSV)
    all_dirs = [path for path in PRED_ROOT.iterdir() if path.is_dir()]
    lower_to_dirs: dict[str, list[Path]] = defaultdict(list)
    for path in all_dirs:
        lower_to_dirs[path.name.lower()].append(path)

    if STAGE_DIR.exists():
        if not REPLACE_STAGE:
            raise RuntimeError(
                f"Stage directory exists: {STAGE_DIR}. "
                "Pass --replace-stage to replace it."
            )
        shutil.rmtree(STAGE_DIR)
    LOCAL_BEST_DIR.mkdir(parents=True, exist_ok=True)

    chosen_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        job_name = row["af3_job_name"]
        job_key = job_name.lower()
        exact_candidates = list(lower_to_dirs.get(job_key, []))
        suffix_candidates = [
            path
            for path in all_dirs
            if re.fullmatch(re.escape(job_key) + r"_\d+", path.name.lower())
        ]
        candidates = sorted(
            {*exact_candidates, *suffix_candidates}, key=lambda p: p.name.lower()
        )
        if not candidates:
            raise RuntimeError(f"No prediction directory found for {job_name}")

        best_choice: dict[str, Any] | None = None
        for pred_dir in candidates:
            summaries = sorted(pred_dir.glob("*summary_confidences_*.json"))
            if not summaries:
                raise RuntimeError(f"No summary JSON files found in {pred_dir}")
            for summary_path in summaries:
                model_match = re.search(r"_([0-9]+)\.json$", summary_path.name)
                if model_match is None:
                    raise RuntimeError(
                        f"Could not parse model index from {summary_path}"
                    )
                model_index = int(model_match.group(1))
                summary = json.loads(summary_path.read_text())
                cif_prefix = re.sub(
                    r"_summary_confidences_\d+\.json$", "", summary_path.name
                )
                cif_path = pred_dir / f"{cif_prefix}_model_{model_index}.cif"
                if not cif_path.exists():
                    raise RuntimeError(f"Missing CIF {cif_path}")
                ranking_score = summary.get("ranking_score")
                ranking_value = (
                    float("-inf") if ranking_score is None else float(ranking_score)
                )
                exact_name_bonus = 1 if pred_dir.name.lower() == job_key else 0
                candidate = {
                    "job_name": job_name,
                    "cluster": row["cluster"],
                    "sample_id": row["sample_id"],
                    "manuscript_group": row["manuscript_group"],
                    "sequence_id": row["sequence_id"],
                    "sequence_length": row["aa_len"],
                    "prediction_dir": str(pred_dir),
                    "prediction_dir_name": pred_dir.name,
                    "model_index": model_index,
                    "ranking_score": "" if ranking_score is None else ranking_score,
                    "ranking_value": ranking_value,
                    "ptm": summary.get("ptm"),
                    "iptm": summary.get("iptm"),
                    "has_clash": summary.get("has_clash"),
                    "summary_path": str(summary_path),
                    "cif_path": str(cif_path),
                    "exact_name_match": exact_name_bonus,
                }
                candidate_rows.append(candidate)
                score_key = (
                    ranking_value,
                    exact_name_bonus,
                    -model_index,
                    pred_dir.name.lower(),
                )
                if best_choice is None or score_key > best_choice["score_key"]:
                    best_choice = {"score_key": score_key, "candidate": candidate}

        if best_choice is None:
            raise RuntimeError(f"Could not choose a best prediction for {job_name}")
        chosen = dict(best_choice["candidate"])
        chosen.pop("ranking_value")
        chosen.pop("exact_name_match")
        chosen["candidate_dir_count"] = len(candidates)
        local_cif_name = (
            f"{chosen['sample_id']}__{safe_name(chosen['sequence_id'])}__"
            f"{safe_name(chosen['prediction_dir_name'])}__model_{chosen['model_index']}.cif"
        )
        local_cif_path = LOCAL_BEST_DIR / local_cif_name
        shutil.copyfile(chosen["cif_path"], local_cif_path)
        chosen["local_cif_path"] = str(local_cif_path)
        chosen["remote_cif_path"] = f"{REMOTE_INPUT_DIR}/{local_cif_name}"
        chosen_rows.append(chosen)

    chosen_rows.sort(key=lambda row: (int(row["cluster"]), row["sample_id"]))
    candidate_rows.sort(
        key=lambda row: (
            int(row["cluster"]),
            row["sample_id"],
            row["prediction_dir_name"],
            row["model_index"],
        )
    )
    return chosen_rows, candidate_rows


def stage_inputs(
    catalog_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> None:
    write_csv(
        STAGE_DIR / "structure_catalog.csv",
        catalog_rows,
        [
            "job_name",
            "cluster",
            "sample_id",
            "manuscript_group",
            "sequence_id",
            "sequence_length",
            "prediction_dir",
            "prediction_dir_name",
            "model_index",
            "ranking_score",
            "ranking_value",
            "ptm",
            "iptm",
            "has_clash",
            "summary_path",
            "cif_path",
            "candidate_dir_count",
            "local_cif_path",
            "remote_cif_path",
        ],
    )
    write_csv(
        STAGE_DIR / "candidate_structure_models.csv",
        candidate_rows,
        [
            "job_name",
            "cluster",
            "sample_id",
            "manuscript_group",
            "sequence_id",
            "sequence_length",
            "prediction_dir",
            "prediction_dir_name",
            "model_index",
            "ranking_score",
            "ranking_value",
            "ptm",
            "iptm",
            "has_clash",
            "summary_path",
            "cif_path",
            "exact_name_match",
        ],
    )


def sync_stage_to_remote() -> None:
    code = f"""
import json
import shutil
from pathlib import Path

root = Path({REMOTE_ROOT!r})
replace = {REPLACE_REMOTE!r}
if root.exists():
    if not replace:
        raise SystemExit(f"Remote directory exists: {{root}}")
    shutil.rmtree(root)
Path({REMOTE_INPUT_DIR!r}).mkdir(parents=True)
Path({REMOTE_WORKDIR!r}).mkdir(parents=True)
print(json.dumps({{"remote_root": str(root), "prepared": True}}))
"""
    json.loads(ssh_python(code))
    run(["rsync", "-a", f"{LOCAL_BEST_DIR}/", f"{REMOTE_HOST}:{REMOTE_INPUT_DIR}/"])


def run_remote_tmalign(catalog_rows: list[dict[str, Any]]) -> dict[str, Any]:
    remote_catalog = [
        {
            "job_name": row["job_name"],
            "cluster": row["cluster"],
            "manuscript_group": row["manuscript_group"],
            "sample_id": row["sample_id"],
            "sequence_id": row["sequence_id"],
            "cif_path": row["remote_cif_path"],
            "prediction_dir_name": row["prediction_dir_name"],
            "model_index": row["model_index"],
            "ranking_score": row["ranking_score"],
        }
        for row in catalog_rows
    ]
    code = f"""
import csv
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

catalog = {json.dumps(remote_catalog)}
workdir = Path({REMOTE_WORKDIR!r})
threads = {THREADS}
tmalign = {TMALIGN!r}

if '/' in tmalign:
    tmalign_path = Path(tmalign)
    if not tmalign_path.is_file():
        raise SystemExit(f'TMalign not found: {{tmalign_path}}')
    tmalign = str(tmalign_path)
else:
    resolved_tmalign = shutil.which(tmalign)
    if resolved_tmalign is None:
        raise SystemExit(f'TMalign not found on PATH: {{tmalign}}')
    tmalign = resolved_tmalign

if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True)

catalog_path = workdir / 'structure_catalog.csv'
fields = [
    'job_name','cluster','manuscript_group','sample_id','sequence_id',
    'cif_path','prediction_dir_name','model_index','ranking_score'
]
with catalog_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(catalog)

pairs = []
for i in range(len(catalog)):
    for j in range(i + 1, len(catalog)):
        pairs.append((catalog[i], catalog[j]))

def parse_output(text):
    aligned_length = None
    rmsd = None
    seq_id = None
    scores = []
    for line in text.splitlines():
        if line.startswith('Aligned length='):
            m = re.search(r'Aligned length=\\s*(\\d+),\\s*RMSD=\\s*([0-9.]+),\\s*Seq_ID=n_identical/n_aligned=\\s*([0-9.]+)', line)
            if m:
                aligned_length = int(m.group(1))
                rmsd = float(m.group(2))
                seq_id = float(m.group(3))
        if line.startswith('TM-score='):
            m = re.search(r'TM-score=\\s*([0-9.]+)', line)
            if m:
                scores.append(float(m.group(1)))
    if len(scores) < 2:
        raise RuntimeError('Could not parse two TM-score lines')
    return aligned_length, rmsd, seq_id, scores[0], scores[1]

def run_pair(item):
    a, b = item
    completed = subprocess.run(
        [tmalign, a['cif_path'], b['cif_path']],
        text=True,
        capture_output=True,
        timeout=300,
    )
    if completed.returncode:
        return {{
            'ok': False,
            'id1': a['job_name'],
            'id2': b['job_name'],
            'returncode': completed.returncode,
            'stdout': completed.stdout,
            'stderr': completed.stderr,
        }}
    aligned_length, rmsd, seq_id, tm1, tm2 = parse_output(completed.stdout)
    return {{
        'ok': True,
        'id1': a['job_name'],
        'cluster1': a['cluster'],
        'group1': a['manuscript_group'],
        'id2': b['job_name'],
        'cluster2': b['cluster'],
        'group2': b['manuscript_group'],
        'score': round((tm1 + tm2) / 2.0, 6),
        'tm_score_chain1': tm1,
        'tm_score_chain2': tm2,
        'rmsd': rmsd,
        'aligned_length': aligned_length,
        'seq_id': seq_id,
    }}

rows = []
failures = []
with ProcessPoolExecutor(max_workers=threads) as pool:
    futures = [pool.submit(run_pair, pair) for pair in pairs]
    for future in as_completed(futures):
        result = future.result()
        if result.pop('ok'):
            rows.append(result)
        else:
            failures.append(result)

rows.sort(key=lambda row: (row['id1'], row['id2']))
failures.sort(key=lambda row: (row['id1'], row['id2']))

pair_path = workdir / 'tmalign_pairwise_unique_pairs.csv'
pair_fields = [
    'id1','cluster1','group1','id2','cluster2','group2',
    'score','tm_score_chain1','tm_score_chain2','rmsd','aligned_length','seq_id'
]
with pair_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=pair_fields)
    writer.writeheader()
    writer.writerows(rows)

failure_path = workdir / 'tmalign_failures.csv'
failure_fields = ['id1','id2','returncode','stderr','stdout']
with failure_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=failure_fields)
    writer.writeheader()
    writer.writerows(failures)

summary = {{
    'remote_workdir': str(workdir),
    'structure_count': len(catalog),
    'expected_unique_pair_count': len(pairs),
    'observed_unique_pair_count': len(rows),
    'failure_count': len(failures),
    'threads': threads,
    'catalog_csv': str(catalog_path),
    'pairwise_csv': str(pair_path),
    'failure_csv': str(failure_path),
}}
(workdir / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
print(json.dumps(summary))
"""
    summary = json.loads(ssh_python(code))
    for remote_name in [
        "structure_catalog.csv",
        "tmalign_pairwise_unique_pairs.csv",
        "tmalign_failures.csv",
        "run_summary.json",
    ]:
        run(
            [
                "scp",
                "-q",
                f"{REMOTE_HOST}:{summary['remote_workdir']}/{remote_name}",
                str(OUTDIR / remote_name),
            ]
        )
    return summary


def aggregate_local(
    catalog_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = read_csv(OUTDIR / "tmalign_pairwise_unique_pairs.csv")

    cluster_group: dict[str, str] = {}
    structure_counts: dict[str, int] = defaultdict(int)
    for row in catalog_rows:
        cluster_group[row["cluster"]] = row["manuscript_group"]
        structure_counts[row["cluster"]] += 1
    clusters = sorted(cluster_group, key=lambda value: int(value))

    by_cluster_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        c1, c2 = row["cluster1"], row["cluster2"]
        key = tuple(sorted((c1, c2), key=lambda value: int(value)))
        by_cluster_pair[key].append(float(row["score"]))

    cluster_pair_rows: list[dict[str, Any]] = []
    for c1 in clusters:
        for c2 in clusters:
            if int(c2) < int(c1):
                continue
            key = tuple(sorted((c1, c2), key=lambda value: int(value)))
            values = by_cluster_pair.get(key, [])
            if not values:
                continue
            cluster_pair_rows.append(
                {
                    "cluster1": c1,
                    "cluster2": c2,
                    "group1": cluster_group[c1],
                    "group2": cluster_group[c2],
                    "same_cluster": str(c1 == c2).lower(),
                    "same_group": str(cluster_group[c1] == cluster_group[c2]).lower(),
                    "n_structures1": structure_counts[c1],
                    "n_structures2": structure_counts[c2],
                    **summarize_numbers(values),
                }
            )

    write_csv(
        OUTDIR / "tmalign_cluster_pair_summary.csv",
        cluster_pair_rows,
        [
            "cluster1",
            "cluster2",
            "group1",
            "group2",
            "same_cluster",
            "same_group",
            "n_structures1",
            "n_structures2",
            "n_pairs",
            "mean_score",
            "median_score",
            "std_score",
            "min_score",
            "q10_score",
            "q25_score",
            "q75_score",
            "q90_score",
            "max_score",
        ],
    )

    mean_lookup = {
        (row["cluster1"], row["cluster2"]): float(row["mean_score"])
        for row in cluster_pair_rows
    }
    for path in [OUTDIR / "tmalign_cluster_mean_tm_matrix.csv"]:
        matrix_rows = []
        for c1 in clusters:
            matrix_row = {"cluster": c1, "group": cluster_group[c1]}
            for c2 in clusters:
                key = (c1, c2) if (c1, c2) in mean_lookup else (c2, c1)
                matrix_row[c2] = mean_lookup.get(key, "")
            matrix_rows.append(matrix_row)
        write_csv(path, matrix_rows, ["cluster", "group", *clusters])

    category_values: dict[str, list[float]] = defaultdict(list)
    for row in cluster_pair_rows:
        if row["same_cluster"] == "true":
            category = "within_cluster"
        elif row["same_group"] == "true":
            category = "between_clusters_same_group"
        else:
            category = "between_groups"
        category_values[category].append(float(row["mean_score"]))

    category_rows = [
        {"category": category, **summarize_numbers(values)}
        for category, values in sorted(category_values.items())
    ]
    write_csv(
        OUTDIR / "tmalign_pair_score_category_summary.csv",
        category_rows,
        [
            "category",
            "n_pairs",
            "mean_score",
            "median_score",
            "std_score",
            "min_score",
            "q10_score",
            "q25_score",
            "q75_score",
            "q90_score",
            "max_score",
        ],
    )

    rng = np.random.default_rng(20260602)
    comparisons = [
        ("within_cluster", "between_clusters_same_group"),
        ("within_cluster", "between_groups"),
        ("between_clusters_same_group", "between_groups"),
    ]
    test_rows = []
    for cat_a, cat_b in comparisons:
        a = np.array(category_values[cat_a], dtype=float)
        b = np.array(category_values[cat_b], dtype=float)
        observed = float(a.mean() - b.mean())
        pooled = np.concatenate([a, b])
        n_a = a.size
        count = 0
        permutations = 20000
        for _ in range(permutations):
            rng.shuffle(pooled)
            diff = float(pooled[:n_a].mean() - pooled[n_a:].mean())
            if abs(diff) >= abs(observed):
                count += 1
        p_value = (count + 1.0) / (permutations + 1.0)
        test_rows.append(
            {
                "category_a": cat_a,
                "category_b": cat_b,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "observed_mean_diff": observed,
                "p_value": p_value,
            }
        )

    indexed = sorted(enumerate(test_rows), key=lambda item: item[1]["p_value"])
    running = 0.0
    adjusted = [0.0] * len(test_rows)
    for rank, (idx, row) in enumerate(indexed):
        running = max(running, min(1.0, row["p_value"] * (len(test_rows) - rank)))
        adjusted[idx] = running
    for row, adj in zip(test_rows, adjusted):
        row["holm_adjusted_p"] = adj
        row["significance"] = (
            "****"
            if adj < 1e-4
            else "***"
            if adj < 1e-3
            else "**"
            if adj < 1e-2
            else "*"
            if adj < 0.05
            else "ns"
        )

    write_csv(
        OUTDIR / "tmalign_pair_score_category_significance_tests.csv",
        test_rows,
        [
            "category_a",
            "category_b",
            "n_a",
            "n_b",
            "mean_a",
            "mean_b",
            "observed_mean_diff",
            "p_value",
            "holm_adjusted_p",
            "significance",
        ],
    )

    (OUTDIR / "tmalign_pair_score_category_significance_summary.json").write_text(
        json.dumps(
            {
                "seed": 20260602,
                "permutations": 20000,
                "category_sizes": {
                    key: len(value) for key, value in category_values.items()
                },
                "tests": test_rows,
            },
            indent=2,
        )
        + "\n"
    )

    return cluster_pair_rows, test_rows


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def y_for(value: float, top: float, plot_h: float) -> float:
    return top + (1.0 - value) * plot_h


def build_svg(
    cluster_pair_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> None:
    values_by_cat: dict[str, list[float]] = {cat: [] for cat in PLOT_ORDER}
    for row in cluster_pair_rows:
        if row["same_cluster"] == "true":
            category = "within_cluster"
        elif row["same_group"] == "true":
            category = "between_clusters_same_group"
        else:
            category = "between_groups"
        values_by_cat[category].append(float(row["mean_score"]))

    arrays_by_cat = {
        cat: np.array(vals, dtype=float) for cat, vals in values_by_cat.items()
    }

    width = 980
    height = 620
    left = 100
    right = 40
    top = 110
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    baseline = top + plot_h
    x_positions = {
        cat: left + plot_w * frac
        for cat, frac in zip(PLOT_ORDER, [0.18, 0.50, 0.82], strict=True)
    }
    colors = {
        "within_cluster": "#0f766e",
        "between_clusters_same_group": "#b45309",
        "between_groups": "#1d4ed8",
    }
    rng = np.random.default_rng(20260602)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="100" y="36" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#111827">TMalign similarity by relationship category</text>',
        '<text x="100" y="58" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">Unit of analysis: cluster-pair mean TM-score, not raw structure pairs. Significance: two-sided permutation test on mean difference with Holm correction.</text>',
        '<text x="100" y="76" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">Dataset: 22 selected no-HMM clusters, using the best available AF3 prediction for each sampled job.</text>',
    ]

    for tick in np.linspace(0, 1, 6):
        y = y_for(float(tick), top, plot_h)
        lines.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6b7280">{tick:.1f}</text>'
        )

    box_w = 86
    for cat in PLOT_ORDER:
        values = arrays_by_cat[cat]
        x = x_positions[cat]
        color = colors[cat]
        q10, q25, q50, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
        mean = float(values.mean())
        lines.extend(
            [
                f'<line x1="{x}" y1="{y_for(float(q10), top, plot_h)}" x2="{x}" y2="{y_for(float(q25), top, plot_h)}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x}" y1="{y_for(float(q75), top, plot_h)}" x2="{x}" y2="{y_for(float(q90), top, plot_h)}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 4}" y1="{y_for(float(q10), top, plot_h)}" x2="{x + box_w / 4}" y2="{y_for(float(q10), top, plot_h)}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 4}" y1="{y_for(float(q90), top, plot_h)}" x2="{x + box_w / 4}" y2="{y_for(float(q90), top, plot_h)}" stroke="{color}" stroke-width="2"/>',
                f'<rect x="{x - box_w / 2}" y="{y_for(float(q75), top, plot_h)}" width="{box_w}" height="{max(2, y_for(float(q25), top, plot_h) - y_for(float(q75), top, plot_h))}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 2}" y1="{y_for(float(q50), top, plot_h)}" x2="{x + box_w / 2}" y2="{y_for(float(q50), top, plot_h)}" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{x}" cy="{y_for(mean, top, plot_h)}" r="4.5" fill="{color}"/>',
            ]
        )
        jitter = rng.uniform(-box_w * 0.42, box_w * 0.42, size=values.size)
        for value, dx in zip(values, jitter, strict=True):
            lines.append(
                f'<circle cx="{x + float(dx)}" cy="{y_for(float(value), top, plot_h)}" r="2.7" fill="{color}" fill-opacity="0.28"/>'
            )
        lines.append(
            f'<text x="{x}" y="{baseline + 28}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape_xml(CATEGORY_LABELS[cat])}</text>'
        )
        lines.append(
            f'<text x="{x}" y="{baseline + 48}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">n={values.size}, mean={values.mean():.3f}</text>'
        )

    bracket_levels = [0.93, 0.975, 0.84]
    for row, y_level in zip(test_rows, bracket_levels, strict=True):
        x1 = x_positions[row["category_a"]]
        x2 = x_positions[row["category_b"]]
        y = y_for(y_level, top, plot_h)
        y_stub = y + 12
        lines.extend(
            [
                f'<line x1="{x1}" y1="{y_stub}" x2="{x1}" y2="{y}" stroke="#111827" stroke-width="1.6"/>',
                f'<line x1="{x2}" y1="{y_stub}" x2="{x2}" y2="{y}" stroke="#111827" stroke-width="1.6"/>',
                f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#111827" stroke-width="1.6"/>',
            ]
        )
        label = f"{row['significance']}  Holm p={fmt_p(row['holm_adjusted_p'])}"
        lines.append(
            f'<text x="{(x1 + x2) / 2}" y="{y - 8}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#111827">{escape_xml(label)}</text>'
        )

    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="#9ca3af" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{baseline}" x2="{left + plot_w}" y2="{baseline}" stroke="#9ca3af" stroke-width="1.2"/>',
            f'<text x="28" y="{top + 40}" transform="rotate(-90 28 {top + 40})" font-size="12" font-family="Arial, sans-serif" fill="#374151">Mean TM-score per cluster-pair</text>',
            "</svg>",
        ]
    )
    (OUTDIR / "tmalign_pair_score_category_significance.svg").write_text(
        "\n".join(lines)
    )


def main() -> None:
    global MANIFEST_CSV, PRED_ROOT, OUTDIR, STAGE_DIR, LOCAL_BEST_DIR
    global REMOTE_HOST, REMOTE_ROOT, REMOTE_INPUT_DIR, REMOTE_WORKDIR
    global TMALIGN, THREADS, REPLACE_STAGE, REPLACE_REMOTE

    args = parse_args()
    MANIFEST_CSV = args.manifest_csv.expanduser().resolve()
    PRED_ROOT = args.prediction_root.expanduser().resolve()
    OUTDIR = args.output_dir.expanduser().resolve()
    STAGE_DIR = args.stage_dir.expanduser().resolve()
    LOCAL_BEST_DIR = STAGE_DIR / "best_models"
    REMOTE_HOST = args.remote_host
    REMOTE_ROOT = args.remote_root.rstrip("/")
    REMOTE_INPUT_DIR = f"{REMOTE_ROOT}/inputs/best_models"
    REMOTE_WORKDIR = f"{REMOTE_ROOT}/results"
    TMALIGN = args.tmalign
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    THREADS = args.threads
    REPLACE_STAGE = args.replace_stage
    REPLACE_REMOTE = args.replace_remote

    OUTDIR.mkdir(parents=True, exist_ok=True)
    chosen_rows, candidate_rows = load_best_prediction_catalog()
    stage_inputs(chosen_rows, candidate_rows)
    sync_stage_to_remote()
    remote_summary = run_remote_tmalign(chosen_rows)
    if remote_summary["failure_count"] != 0:
        raise RuntimeError(
            f"TMalign produced failures; see {OUTDIR / 'tmalign_failures.csv'}"
        )
    if (
        remote_summary["observed_unique_pair_count"]
        != remote_summary["expected_unique_pair_count"]
    ):
        raise RuntimeError(
            "TMalign pair count mismatch: "
            f"{remote_summary['observed_unique_pair_count']} vs {remote_summary['expected_unique_pair_count']}"
        )
    cluster_pair_rows, test_rows = aggregate_local(chosen_rows)
    build_svg(cluster_pair_rows, test_rows)

    summary = {
        "manifest_csv": str(MANIFEST_CSV),
        "prediction_root": str(PRED_ROOT),
        "chosen_structure_count": len(chosen_rows),
        "candidate_model_count": len(candidate_rows),
        "remote_root": REMOTE_ROOT,
        "remote_summary": remote_summary,
        "outputs": {
            "catalog_csv": str(STAGE_DIR / "structure_catalog.csv"),
            "candidate_models_csv": str(
                STAGE_DIR / "candidate_structure_models.csv"
            ),
            "pairwise_csv": str(OUTDIR / "tmalign_pairwise_unique_pairs.csv"),
            "cluster_pair_summary_csv": str(
                OUTDIR / "tmalign_cluster_pair_summary.csv"
            ),
            "category_summary_csv": str(
                OUTDIR / "tmalign_pair_score_category_summary.csv"
            ),
            "significance_tests_csv": str(
                OUTDIR / "tmalign_pair_score_category_significance_tests.csv"
            ),
            "significance_summary_json": str(
                OUTDIR / "tmalign_pair_score_category_significance_summary.json"
            ),
            "significance_svg": str(
                OUTDIR / "tmalign_pair_score_category_significance.svg"
            ),
        },
    }
    (OUTDIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
