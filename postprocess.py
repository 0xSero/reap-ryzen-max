"""Recompute REAP / ean_mean from sums for any expert-saliency.json.

Use this if you have an ``expert-saliency.json`` written by an older REAP observer that
accumulated *per-batch means* into ``reap`` / ``ean_mean`` (a common bug — including in
earlier versions of this very repo). The underlying ``weighted_ean_sum`` and ``ean_sum``
fields are correct, so the fix is a one-line division.

Definitions (Cerebras schema, what downstream planners expect):

    reap[i]     = weighted_ean_sum[i] / max(1, expert_frequency[i])
    ean_mean[i] = ean_sum[i]          / max(1, expert_frequency[i])

Usage:

    python postprocess.py path/to/expert-saliency.json --output path/to/expert-saliency-fixed.json
    python postprocess.py path/to/expert-saliency.json --in-place

If the input has both ``layers`` (dict) and per-layer ``weighted_ean_sum`` / ``ean_sum``
fields, this script overwrites ``reap`` and ``ean_mean`` in place and writes the result.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def fix_layer(layer: dict) -> dict:
    ef = layer.get("expert_frequency")
    wes = layer.get("weighted_ean_sum")
    es = layer.get("ean_sum")
    if not (ef and wes and es):
        raise KeyError(
            "Layer is missing one of: expert_frequency / weighted_ean_sum / ean_sum. "
            "This script needs the raw sums; rerun your observation with reap_observer.py."
        )
    n = len(ef)
    reap = [(wes[i] / ef[i]) if ef[i] > 0 else 0.0 for i in range(n)]
    ean_mean = [(es[i] / ef[i]) if ef[i] > 0 else 0.0 for i in range(n)]
    layer["reap"] = reap
    layer["ean_mean"] = ean_mean
    return layer


def fix_document(doc: dict) -> dict:
    if "layers" in doc and isinstance(doc["layers"], dict):
        doc["layers"] = {k: fix_layer(v) for k, v in doc["layers"].items()}
    elif all(k.isdigit() for k in doc.keys()):
        # bare {layer_idx: layer_dict, ...}
        doc = {k: fix_layer(v) for k, v in doc.items()}
    else:
        raise ValueError(
            "Could not find a 'layers' dict or a bare layer-index dict at the document root."
        )
    return doc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path, help="expert-saliency.json to fix")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--output", type=pathlib.Path, help="write corrected JSON here")
    g.add_argument("--in-place", action="store_true", help="overwrite the input file")
    args = p.parse_args()

    doc = json.loads(args.input.read_text())
    fixed = fix_document(doc)

    out_path = args.input if args.in_place else args.output
    out_path.write_text(json.dumps(fixed, indent=2))
    print(f"wrote corrected REAP/ean_mean to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
