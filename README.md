# reap-ryzen-max

A small overlay for running [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
on AMD's Ryzen AI MAX+ 395 ("Strix Halo") with the gfx1151 iGPU and 128 GB
unified memory. The actual REAP science lives in the upstream repo. This one
just gives you the install incantations and the Nemotron-3 patch that don't
exist there yet.

## What you get

- `bootstrap.sh` — installs PyTorch from TheRock's gfx1151 nightly index,
  clones Cerebras REAP, and snapshots Nemotron-3 with the modeling patches
  applied.
- `run.sh` — convenience wrapper around `experiments/pruning-cli.sh` with
  defaults that work on Strix Halo (CPU device map for the 30B model, modest
  sample budget).
- A pointer to my upstream PR so you can see what's actually changed in
  Cerebras REAP.

## What's broken upstream right now

Nemotron-3 / Nemotron-H out of the box doesn't profile cleanly under Cerebras
REAP. Three reasons:

1. `NemotronHMOE.forward` returns only the combined hidden states (no
   `router_logits`), which trips the standard observer's tuple-unpack contract.
2. The HF modeling file hard-imports a Triton kernel from `mamba_ssm`, so it
   fails to import on CPU and AMD ROCm hosts.
3. `NemotronHBlock.forward` wraps work in `torch.cuda.stream(...)`, which is
   invalid off-CUDA.

I sent a PR to Cerebras adding the patched modeling file, the registry
entries, and the AMD/Strix Halo install notes:
**[CerebrasResearch/reap#22](https://github.com/CerebrasResearch/reap/pull/22)**.
Until it's merged, `bootstrap.sh` checks out my fork's branch directly.

## Hardware assumed

- Framework desktop with AMD Ryzen AI MAX+ 395 (gfx1151 iGPU)
- 128 GB unified memory
- Linux, ROCm-capable distro (tested on Fedora 43)
- `~/.cache/huggingface` on a disk with at least ~70 GB free for the
  Nemotron-3 30B-A3B snapshot

## Quick start

```bash
git clone https://github.com/0xSero/reap-ryzen-max.git
cd reap-ryzen-max
bash bootstrap.sh
bash run.sh
```

`bootstrap.sh` does, in order:

1. `pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/`
2. `git clone https://github.com/0xSero/reap.git -b nemotron3-and-strix-halo` (the PR branch)
3. `pip install -e ./reap`
4. `python ./reap/scripts/patch_nemotron_h.py` (downloads the model, swaps in the patched modeling file)

Once Cerebras merges the PR, edit `bootstrap.sh` to clone `CerebrasResearch/reap`
on `main` instead.

## Why CPU device map on Strix Halo

`run.sh` uses `--device-map cpu` for Nemotron-3 30B-A3B (~60 GiB at bf16).
The iGPU and CPU share the same DRAM, and the iGPU compiler doesn't reliably
handle 60 GB allocations. You'll get more deterministic throughput sticking to
the CPU path. The smaller MoEs (Mixtral, Qwen3-30B-A3B) work fine with
`--device-map auto`.

## Other models

This overlay isn't Nemotron-specific past `bootstrap.sh`. Anything Cerebras
REAP supports — Mixtral, Qwen3-MoE, DeepSeek V2/V3, GLM-4.5, Llama-4, ERNIE
4.5 — runs the same way through `experiments/pruning-cli.sh` once
`bootstrap.sh` has set up the wheels and the cache. Only the Nemotron snapshot
step is specific.

## Outputs

Cerebras REAP writes its observer state and pruned-model artifacts under its
own `experiments/output/` tree. Read the upstream README for the canonical
output layout. The schema (`expert_frequency`, `weighted_ean_sum`, `ean_sum`,
`reap`, `max_activations`, etc.) is what any downstream pruner expects.

## Citation

REAP: Lasby, Lazarevich, Sinnadurai, Lie, Ioannou, Thangarasa,
"REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression," ICLR 2026.

## License

MIT for the overlay scripts here. The patched `modeling_nemotron_h.py` lives
in my upstream PR and inherits NVIDIA's Apache-2.0 license. Cerebras REAP is
Apache-2.0.
