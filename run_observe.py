"""Generic REAP observation runner for HuggingFace MoE causal LMs.

Usage (any MoE model from HF Hub with no shared expert):

    python run_observe.py --model-id Qwen/Qwen3-30B-A3B \
        --moe-spec qwen3_moe \
        --max-samples 256 --max-tokens 2048 \
        --device-map auto --pack-samples

For Nemotron-3 / DeepSeek-V3 / Mixtral:

    python run_observe.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --moe-spec nemotron_h ...
    python run_observe.py --model-id deepseek-ai/DeepSeek-V3 --moe-spec deepseek_v3 ...
    python run_observe.py --model-id mistralai/Mixtral-8x7B-v0.1 --moe-spec mixtral ...

Or let the runner guess:

    python run_observe.py --model-id <any> --moe-spec auto ...

Outputs go to ``out/run-<UTC>/``:

    manifest.json          run config
    status.json            live progress
    sample-summary.jsonl   per-sample wall-clock timing
    expert-saliency.json   per-layer per-expert sums + derived means (REAP signal)
    observer-state.pt      raw torch state dict (resumable)
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import random
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reap_observer import MoEObserver, BUILTIN_SPECS, resolve_spec  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reap.run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic REAP observer runner.")
    p.add_argument("--model-id", required=True, help="HuggingFace model id, e.g. Qwen/Qwen3-30B-A3B")
    p.add_argument(
        "--moe-spec", default="auto",
        choices=["auto", *BUILTIN_SPECS.keys()],
        help="Which MoE block layout to assume. 'auto' tries to detect.",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto", help='"auto", "cpu", "cuda", or a JSON map.')
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--pack-samples", action="store_true", default=True)
    p.add_argument("--no-pack-samples", dest="pack_samples", action="store_false")
    p.add_argument("--checkpoint-every-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--calibration-dataset", default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset id with a 'text' field (or similar).",
    )
    p.add_argument("--calibration-split", default="train")
    p.add_argument("--calibration-config", default="sample-10BT", help="Optional ds config name (or empty).")
    p.add_argument("--text-field", default="text")
    p.add_argument("--out-root", default=str(ROOT / "out"))
    p.add_argument("--num-routed-experts", type=int, default=None,
                   help="Override; otherwise read from config (n_routed_experts / num_experts / num_local_experts).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override; otherwise read from config (num_experts_per_tok).")
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _infer_moe_dims(cfg, args: argparse.Namespace) -> tuple[int, int]:
    """Pull (num_routed_experts, top_k) from the config, with overrides."""
    n_routed = args.num_routed_experts
    if n_routed is None:
        for attr in ("n_routed_experts", "num_experts", "num_local_experts", "num_routed_experts"):
            if hasattr(cfg, attr):
                n_routed = int(getattr(cfg, attr))
                break
    if n_routed is None:
        raise ValueError("Could not infer num_routed_experts from config; pass --num-routed-experts.")

    top_k = args.top_k
    if top_k is None:
        for attr in ("num_experts_per_tok", "moe_top_k", "top_k"):
            if hasattr(cfg, attr):
                top_k = int(getattr(cfg, attr))
                break
    if top_k is None:
        raise ValueError("Could not infer top_k from config; pass --top-k.")

    return n_routed, top_k


def load_calibration(args: argparse.Namespace, tokenizer):
    """Load and pack calibration text into ``--max-tokens``-token windows."""
    from datasets import load_dataset

    log.info("loading calibration dataset %s (config=%r split=%r)",
             args.calibration_dataset, args.calibration_config, args.calibration_split)
    kwargs: dict = {"split": args.calibration_split, "streaming": True}
    if args.calibration_config:
        kwargs["name"] = args.calibration_config
    ds = load_dataset(args.calibration_dataset, **kwargs)

    samples: list[str] = []
    for ex in ds:
        t = ex.get(args.text_field) or ex.get("text") or ex.get("content") or ex.get("input")
        if isinstance(t, list):
            t = "\n".join(str(x) for x in t)
        if isinstance(t, str) and len(t) > 32:
            samples.append(t)
        if len(samples) >= args.max_samples * 8:
            break

    random.Random(args.seed).shuffle(samples)
    log.info("collected %d raw text samples", len(samples))

    eos = tokenizer.eos_token_id or tokenizer.bos_token_id or 0
    packed: list[torch.Tensor] = []

    if args.pack_samples:
        buf: list[int] = []
        for s in samples:
            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            ids.append(eos)
            buf.extend(ids)
            while len(buf) >= args.max_tokens:
                packed.append(torch.tensor(buf[: args.max_tokens], dtype=torch.long))
                buf = buf[args.max_tokens:]
                if len(packed) >= args.max_samples:
                    break
            if len(packed) >= args.max_samples:
                break
    else:
        for s in samples:
            ids = tokenizer(s, truncation=True, max_length=args.max_tokens)["input_ids"]
            packed.append(torch.tensor(ids, dtype=torch.long))
            if len(packed) >= args.max_samples:
                break

    log.info("packed %d sequences of up to %d tokens", len(packed), args.max_tokens)
    return packed


def main() -> int:
    args = parse_args()
    if args.min_samples is None:
        args.min_samples = args.max_samples
    torch.manual_seed(args.seed)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_root) / f"run-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output dir: %s", out_dir)

    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    log.info("config_class=%s num_hidden_layers=%d", type(cfg).__name__, cfg.num_hidden_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    log.info("loading model dtype=%s device_map=%s ...", args.dtype, args.device_map)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("model loaded in %.1fs", time.time() - t0)

    n_routed, top_k = _infer_moe_dims(cfg, args)
    spec = resolve_spec(args.moe_spec, model)

    manifest = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_routed_experts": int(n_routed),
        "top_k": int(top_k),
        "moe_spec": spec.name,
        "calibration_dataset": args.calibration_dataset,
        "calibration_config": args.calibration_config,
        "calibration_split": args.calibration_split,
        "max_samples": int(args.max_samples),
        "max_tokens": int(args.max_tokens),
        "pack_samples": bool(args.pack_samples),
        "seed": int(args.seed),
        "started_at": ts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    calib = load_calibration(args, tokenizer)
    if len(calib) < args.min_samples:
        (out_dir / "status.json").write_text(json.dumps({
            "state": "stopped",
            "blocker": f"calibration produced {len(calib)} sequences < min_samples={args.min_samples}",
        }, indent=2))
        log.error("calibration too small: %d < %d", len(calib), args.min_samples)
        return 2

    observer = MoEObserver(model=model, spec=spec, num_routed_experts=n_routed, top_k=top_k)

    sample_path = out_dir / "sample-summary.jsonl"
    sample_f = open(sample_path, "a", buffering=1)
    try:
        with torch.no_grad():
            for i, ids in enumerate(calib[: args.max_samples]):
                ids = ids.unsqueeze(0).to(next(model.parameters()).device)
                t0 = time.time()
                model(input_ids=ids)
                dt = time.time() - t0
                sample_f.write(json.dumps({"i": i, "tokens": int(ids.numel()), "secs": round(dt, 3)}) + "\n")

                if (i + 1) % max(1, args.checkpoint_every_samples) == 0 or (i + 1) == args.max_samples:
                    observer.save_state(out_dir / "observer-state.pt")
                    snap = observer.report_state()
                    (out_dir / "expert-saliency.json").write_text(json.dumps({
                        "model_id": args.model_id,
                        "moe_spec": spec.name,
                        "num_routed_experts_per_layer": int(n_routed),
                        "top_k": int(top_k),
                        "calibration": {
                            "dataset": args.calibration_dataset,
                            "config": args.calibration_config,
                            "n_samples": int(i + 1),
                            "max_tokens": int(args.max_tokens),
                            "packed": bool(args.pack_samples),
                        },
                        "layers": {str(k): v for k, v in snap.items()},
                    }, indent=2))
                    (out_dir / "status.json").write_text(json.dumps({
                        "state": "running",
                        "samples_done": i + 1,
                        "layers_seen": sorted(snap.keys()),
                        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    }, indent=2))
                    log.info("ckpt sample %d/%d layers=%d dt=%.2fs", i + 1, args.max_samples, len(snap), dt)

        observer.save_state(out_dir / "observer-state.pt")
        snap = observer.report_state()
        (out_dir / "status.json").write_text(json.dumps({
            "state": "completed",
            "samples_done": len(calib),
            "layers_seen": sorted(snap.keys()),
        }, indent=2))
    finally:
        sample_f.close()
        observer.close_hooks()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
