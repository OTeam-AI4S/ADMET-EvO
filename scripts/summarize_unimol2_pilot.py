"""Build a compact Markdown report for the six-task Uni-Mol2 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/unimol2-pilot-v22"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        champion = result["champion"]
        champion_score = result["summary"][champion]
        fine_score = result["summary"]["finetuned_unimol2"]
        tabular = {
            name: value
            for name, value in result["summary"].items()
            if name.startswith("2d_")
        }
        direction = -1 if result["metric"] == "mae" else 1
        best_2d, best_2d_score = max(
            tabular.items(), key=lambda item: direction * item[1]["mean"]
        )
        rows.append(
            {
                "task": result["task"],
                "metric": result["metric"],
                "champion": champion,
                "champion_score": champion_score,
                "unimol2": fine_score,
                "best_2d": best_2d,
                "best_2d_score": best_2d_score,
            }
        )

    lines = [
        "# Uni-Mol2 V2.2 six-task validation pilot",
        "",
        (
            "Protocol: five TDC default scaffold validation seeds; five inner scaffold OOF "
            "folds; Uni-Mol2 84M full fine-tuning for 20 epochs. Test labels were not used."
        ),
        "",
        "| Task | Metric | Validation champion | Champion score | Fine-tuned Uni-Mol2 | Best 2D |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        score = row["champion_score"]
        fine = row["unimol2"]
        best_2d = row["best_2d_score"]
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['champion']} | "
            f"{score['mean']:.4f} +/- {score['std']:.4f} | "
            f"{fine['mean']:.4f} +/- {fine['std']:.4f} | "
            f"{row['best_2d']} {best_2d['mean']:.4f} +/- {best_2d['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Conclusion: Uni-Mol2 is a task-specific expert, not a universal replacement. "
                "The validation router selects direct Uni-Mol2, 2D models, OOF stacking, and "
                "rank ensembles on different endpoints."
            ),
            "",
        ]
    )
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.root / "summary.json").write_text(
        json.dumps({"protocol_test_labels_used": False, "tasks": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
