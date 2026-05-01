"""Apply REAP-PATCH stanzas to HuggingFace `trust_remote_code` modeling files.

Many modern MoE models ship modeling code that hard-imports CUDA-only kernels
(``mamba_ssm.ops.triton.layer_norm.rmsnorm_fn``) or wraps work in a
``torch.cuda.stream(...)`` context. Both crash on CPU-only or non-NVIDIA GPU
(AMD ROCm) hosts. This script idempotently patches the on-disk cached modeling
files to fall back to pure-torch equivalents when the CUDA path is unavailable.

It applies the patches in-place to the file under
``~/.cache/huggingface/modules/transformers_modules/<model>/modeling_*.py``,
which is what ``transformers`` actually imports. Re-running is a no-op.

Currently handles:
  * ``modeling_nemotron_h.py``   (Nemotron-3 / Nemotron-H family)

To add a new model, add a ``PATCH`` entry mapping a regex anchor to the
replacement block. Patches are guarded with a ``# REAP-PATCH`` marker so the
script can detect already-applied edits.

Usage:

    python patch_modeling.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

If the model's modeling file isn't in the cache yet, run any
``AutoModel.from_pretrained(..., trust_remote_code=True)`` first to populate it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from dataclasses import dataclass


@dataclass
class Patch:
    name: str
    marker: str
    pattern: str  # regex, anchored
    replacement: str


# ---- Patches ----

NEMOTRON_PATCHES: list[Patch] = [
    Patch(
        name="rmsnorm_fn pure-torch fallback",
        marker="REAP-PATCH:rmsnorm_fn",
        pattern=r"from mamba_ssm\.ops\.triton\.layer_norm import (rmsnorm_fn[^\n]*)",
        replacement=(
            "# REAP-PATCH:rmsnorm_fn  pure-torch fallback when mamba_ssm/triton is unavailable\n"
            "try:\n"
            "    from mamba_ssm.ops.triton.layer_norm import \\1\n"
            "except Exception:\n"
            "    import torch\n"
            "    def rmsnorm_fn(x, weight, bias=None, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6):\n"
            "        if residual is not None:\n"
            "            x = x + residual\n"
            "            if residual_in_fp32:\n"
            "                x = x.to(torch.float32)\n"
            "        var = x.pow(2).mean(-1, keepdim=True)\n"
            "        x_norm = x * torch.rsqrt(var + eps)\n"
            "        out = x_norm.to(weight.dtype) * weight\n"
            "        if bias is not None:\n"
            "            out = out + bias\n"
            "        return (out, x) if prenorm else out\n"
        ),
    ),
    Patch(
        name="torch.cuda.stream nullcontext guard",
        marker="REAP-PATCH:cuda_stream",
        pattern=r"with torch\.cuda\.stream\(([^)]+)\):",
        replacement=(
            "# REAP-PATCH:cuda_stream  CPU-safe guard\n"
            "from contextlib import nullcontext as _reap_nullcontext\n"
            "_reap_stream_ctx = torch.cuda.stream(\\1) if torch.cuda.is_available() else _reap_nullcontext()\n"
            "with _reap_stream_ctx:"
        ),
    ),
]

PATCH_REGISTRY: dict[str, list[Patch]] = {
    "modeling_nemotron_h.py": NEMOTRON_PATCHES,
}


def cache_root() -> pathlib.Path:
    base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return pathlib.Path(base) / "modules" / "transformers_modules"


def find_modeling_files(model_id: str) -> list[pathlib.Path]:
    """Locate cached modeling_*.py files for a given model id."""
    root = cache_root()
    if not root.exists():
        return []
    # HF rewrites the model id slug as <user>--<repo> or just <repo>
    candidates = list(root.rglob("modeling_*.py"))
    short = model_id.replace("/", "--")
    namebits = [model_id.split("/")[-1].lower(), short.lower()]
    matches = [p for p in candidates if any(bit in str(p).lower() for bit in namebits)]
    return matches or candidates


def apply_patch(text: str, patch: Patch) -> tuple[str, bool]:
    """Apply one patch idempotently. Returns (new_text, applied)."""
    if patch.marker in text:
        return text, False
    if not re.search(patch.pattern, text):
        return text, False
    new_text = re.sub(patch.pattern, patch.replacement, text, count=1)
    return new_text, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HuggingFace model id, e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change but do not write.")
    args = ap.parse_args()

    files = find_modeling_files(args.model_id)
    if not files:
        print(
            f"No cached modeling files found under {cache_root()}.\n"
            "Run AutoModel.from_pretrained(..., trust_remote_code=True) once to populate the cache.",
            file=sys.stderr,
        )
        return 1

    any_changed = False
    for path in files:
        patches = PATCH_REGISTRY.get(path.name)
        if not patches:
            continue
        text = path.read_text()
        new_text = text
        applied: list[str] = []
        for p in patches:
            new_text, ok = apply_patch(new_text, p)
            if ok:
                applied.append(p.name)
        if applied:
            any_changed = True
            print(f"{'(dry) ' if args.dry_run else ''}patched {path}:")
            for a in applied:
                print(f"    - {a}")
            if not args.dry_run:
                backup = path.with_suffix(path.suffix + ".reap.bak")
                if not backup.exists():
                    backup.write_text(text)
                path.write_text(new_text)
        else:
            print(f"no changes needed for {path}")

    if not any_changed:
        print("Nothing to do; modeling files already patched (or no patches registered for this model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
