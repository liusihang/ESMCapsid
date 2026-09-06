from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiment_tools.layer_selection_utils import choose_stage_layers
except ImportError:  # pragma: no cover - remote flat-script fallback
    from layer_selection_utils import choose_stage_layers


def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--selected-layers-json", required=True)
    parser.add_argument("--early-fraction", type=float, default=0.5)
    parser.add_argument("--late-fraction", type=float, default=0.5)
    parser.add_argument("--early-metric", default="recall_at_fpr_0_001")
    parser.add_argument("--late-metric", default="precision_at_fpr_0_0001")
    args = parser.parse_args()

    scan_root = Path(args.scan_root)
    rows = []
    for summary_path in sorted(scan_root.glob("layer_*/summary.json")):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if not rows:
        raise FileNotFoundError(f"no layer summaries found under {scan_root}")

    summary_df = pd.DataFrame(rows).sort_values("layer")
    summary_df.to_csv(scan_root / "layer_summary.csv", index=False)

    selection = choose_stage_layers(
        rows,
        early_fraction=args.early_fraction,
        late_fraction=args.late_fraction,
        early_metric=args.early_metric,
        late_metric=args.late_metric,
    )
    selection_payload = {
        "model_name": args.model_name,
        "scan_root": str(scan_root),
        "early_fraction": args.early_fraction,
        "late_fraction": args.late_fraction,
        "early_metric": args.early_metric,
        "late_metric": args.late_metric,
        "early_layer": int(selection["early"]["layer"]),
        "late_layer": int(selection["late"]["layer"]),
        "selection": selection,
    }

    selected_json = Path(args.selected_layers_json)
    selected_json.parent.mkdir(parents=True, exist_ok=True)
    selected_json.write_text(
        json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
    )

    pd.DataFrame(
        [
            {"stage": "early", **selection["early"]},
            {"stage": "late", **selection["late"]},
        ]
    ).to_csv(scan_root / "selected_layers.csv", index=False)


if __name__ == "__main__":
    main()
