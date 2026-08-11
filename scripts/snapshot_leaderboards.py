"""Snapshot all official TDC ADMET leaderboard tables."""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BASE = "https://tdcommons.ai"
START = f"{BASE}/benchmark/admet_group/01caco2/"


def parse_score(value: object) -> tuple[float, float | None]:
    numbers = re.findall(r"[-+]?\d*\.?\d+", str(value))
    if not numbers:
        raise ValueError(f"Could not parse leaderboard score: {value!r}")
    return float(numbers[0]), float(numbers[1]) if len(numbers) > 1 else None


def snapshot() -> dict:
    first = requests.get(START, timeout=30)
    first.raise_for_status()
    paths = sorted(
        {
            value
            for value in re.findall(
                r"""href=["']([^"']*/benchmark/admet_group/\d{2}[^/"']+/)["']""",
                first.text,
            )
            if value.startswith("/")
        }
    )
    if len(paths) != 22:
        raise RuntimeError(f"Expected 22 leaderboard pages, found {len(paths)}")
    pages = {}
    for path in paths:
        url = urljoin(BASE, path)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        if len(tables) < 2:
            raise RuntimeError(f"No leaderboard table at {url}")
        summary, leaderboard = tables[0], tables[1]
        dataset = str(summary.iloc[0]["Dataset"]).removeprefix("TDC.")
        metric = str(summary.iloc[0]["Metric"]).lower()
        score_column = leaderboard.columns[-1]
        entries = []
        for _, row in leaderboard.iterrows():
            mean, std = parse_score(row[score_column])
            entries.append(
                {
                    "rank": int(row["Rank"]),
                    "model": str(row["Model"]),
                    "mean": mean,
                    "std": std,
                }
            )
        pages[dataset] = {
            "url": url,
            "metric": metric,
            "direction": "minimize" if metric == "mae" else "maximize",
            "size": int(str(summary.iloc[0]["Size"]).replace(",", "")),
            "entries": entries,
        }
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": START,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/leaderboard-v21/leaderboard_snapshot.json"),
    )
    args = parser.parse_args()
    payload = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"captured_at": payload["captured_at"], "tasks": len(payload["pages"])}))


if __name__ == "__main__":
    main()
