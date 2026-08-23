"""Evaluate protocol v2 including the serialization-adapter binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_openadmet_external_propensity_gate import (
    _load,
    _protocol,
    evaluate as evaluate_v1,
    render_markdown,
)


def evaluate(protocol: dict, result: dict) -> dict:
    payload = evaluate_v1(protocol, result)
    adapter_check = {
        "name": "compatibility_adapter_binding",
        "observed": result.get("compatibility_adapter_sha256"),
        "requirement": protocol["compatibility_adapter_sha256"],
        "passed": (
            result.get("compatibility_adapter_sha256")
            == protocol["compatibility_adapter_sha256"]
            and result.get("aborted_predecessor_protocol_sha256")
            == protocol["aborted_predecessor_protocol_sha256"]
            and protocol.get("scientific_design_changed_from_predecessor") is False
        ),
    }
    payload["version"] = "openadmet-external-propensity-gate-v2"
    payload["checks"].insert(1, adapter_check)
    payload["promotion_passed"] = all(item["passed"] for item in payload["checks"])
    payload["external_data_action_admitted"] = payload["promotion_passed"]
    payload["interpretation"] = (
        "The auxiliary Veith propensity bridge passed the frozen v2 training-only gate."
        if payload["promotion_passed"]
        else "The auxiliary Veith propensity bridge failed the frozen v2 gate and remains excluded from the blind recipe."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(_protocol(args.protocol), _load(args.result))
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
