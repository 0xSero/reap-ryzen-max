# reap-ryzen-max

A correct, generic [REAP](https://arxiv.org/abs/2510.13999) (Routed-Expert
Activation Profiling) observer for HuggingFace Mixture-of-Experts language
models, designed to run on commodity hardware including AMD's Ryzen AI MAX+
395 ("Strix Halo") with 128 GB unified memory and the gfx1151 iGPU under
ROCm.

REAP records, per MoE layer and per *routed* expert, the signals a downstream
planner needs to safely prune or merge experts:

| field                            | meaning                                        |
| -------------------------------- | ---------------------------------------------- |
| `expert_frequency[i]`            | # of (token, top-k slot) pairs that selected i |
| `weighted_expert_frequency_sum[i]` | Σ renormalized router weight to expert i     |
| `ean_sum[i]`                     | Σ ‖expert_i(x)‖₂ over tokens that chose i      |
| `weighted_ean_sum[i]`            | Σ ‖expert_i(x)‖₂ × renorm_weight (REAP signal) |
| `max_activations[i]`             | max ‖expert_i(x)‖∞ seen so far                 |
| `ean_mean[i]`                    | `ean_sum[i] / expert_frequency[i]`             |
| `reap[i]`                        | `weighted_ean_sum[i] / expert_frequency[i]`    |

The shared / always-on expert (if the architecture has one) is structurally
excluded — only experts exposed under the configured `experts_attr` are
profiled.

## Why this repo exists

The Cerebras REAP reference observer assumes the MoE block returns a 2-tuple
`(hidden_states, router_logits)` with raw, softmax-able logits. Several modern
MoE stacks — Nemotron-3 Nano, DeepSeek V3, Qwen3-MoE, Mixtral — return only
the combined hidden states from their forward, with sigmoid + grouped-topk +
renormalization happening *inside* the gate. The default observer either
crashes or silently records nothing.

This repo also fixes a subtle accumulator bug present in earlier custom
observers, including the one we ran on Nemotron-3 Nano:

> The buggy code stored the *per-batch mean* of `‖y‖ × w` into a `reap_sum`
> accumulator, then divided by the *total* `expert_frequency` at report time.
> The result systematically dampens hot experts more than cold ones, flattens
> the saliency landscape, and distorts mid-band rankings. Top-end and
> bottom-end ordering is mostly preserved; mid-band rankings are not.

The fix is a one-line change: accumulate `.sum()` not `.mean()`, and divide by
frequency exactly once. See the comment block in `reap_observer.py`.

`postprocess.py` lets anyone with an existing `expert-saliency.json` from the
buggy version recover correct values, since the underlying `weighted_ean_sum`
and `ean_sum` accumulators were always summed correctly.

## Files

| file                  | purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `reap_observer.py`    | Generic `MoEObserver` + `MoESpec` + 4 built-in specs |
| `run_observe.py`      | CLI runner: load model, build calib, observe, dump   |
| `postprocess.py`      | Fix `expert-saliency.json` from buggy older runs     |
| `patch_modeling.py`   | Idempotently patch HF cached modeling code for CPU/AMD |
| `requirements.txt`    | torch, transformers, datasets, accelerate, safetensors |
| `README.md`           | this file                                            |

## Built-in MoE specs

| `--moe-spec`   | matches                                              |
| -------------- | ---------------------------------------------------- |
| `nemotron_h`   | `NemotronHMOE` (Nemotron-3 / Nemotron-H family)      |
| `deepseek_v3`  | `DeepseekV3MoE`, `DeepseekV2MoE`                     |
| `qwen3_moe`    | `Qwen3MoeSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`   |
| `mixtral`      | `MixtralSparseMoeBlock`                              |
| `auto`         | best-effort heuristic                                |

To support a new architecture, write a 4-line `MoESpec` and pass it in:

```python
from reap_observer import MoESpec, MoEObserver

spec = MoESpec(
    name="my_moe",
    block_classes=("MyMoeBlock",),
    experts_attr="experts",   # nn.ModuleList of routed experts
    gate_attr="router",       # gate/router module name
    gate_call="logits",       # "topk" if gate returns (idx, w); "logits" for raw logits
)
obs = MoEObserver(model, spec, num_routed_experts=128, top_k=8)
```

## Quick start

```bash
# 1. Install deps (see requirements.txt for AMD/ROCm + Mamba-2 notes)
pip install -r requirements.txt

# 2. (Optional) Patch HF cached modeling code so it runs on CPU / AMD.
#    Skip this on a CUDA box with mamba-ssm installed.
python patch_modeling.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# 3. Observe. Defaults: bf16, device_map=auto, fineweb-edu calibration,
#    256 packed sequences of 2048 tokens, checkpoint every 32 samples.
python run_observe.py \
    --model-id Qwen/Qwen3-30B-A3B \
    --moe-spec qwen3_moe \
    --max-samples 256 --max-tokens 2048

# 4. Outputs land in ./out/run-<UTC>/:
#       manifest.json  status.json  sample-summary.jsonl
#       expert-saliency.json  observer-state.pt

# 5. If you have an old, buggy expert-saliency.json from a previous tool:
python postprocess.py path/to/expert-saliency.json --output fixed.json
```

## Hardware notes

### AMD Ryzen AI MAX+ 395 (Strix Halo)

- 128 GB UMA — fits any single-mode MoE up to ~60 B params at bf16 with
  headroom (Nemotron-3 Nano 30B-A3B uses ~60 GiB resident).
- gfx1151 iGPU is **not** in stock PyTorch ROCm 6.4 wheels. You will get
  `HIP error: no kernel image is available` until you install TheRock's
  nightly index:

  ```
  pip install --pre torch \
      --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
  ```

- The runner defaults to `--device-map auto`; on Strix Halo, `--device-map cpu`
  is often more reliable for very large MoEs because the iGPU and CPU share
  the same DRAM and the iGPU compiler doesn't always like 60 GB allocations.

### NVIDIA / CUDA

- Models with Mamba-2 blocks (Nemotron-H, etc.) want
  `pip install mamba-ssm causal-conv1d`. Without it, `patch_modeling.py`
  installs a pure-torch fallback that's much slower but correct.

### Memory

The runner loads the **full model into RAM/VRAM** at once. There is no layer
streaming. Plan accordingly: 30B bf16 ≈ 60 GiB, plus activations and HF cache.

## Output schema

`expert-saliency.json` (top level):

```json
{
  "model_id": "...",
  "moe_spec": "qwen3_moe",
  "num_routed_experts_per_layer": 128,
  "top_k": 8,
  "calibration": {"dataset": "...", "n_samples": 256, "max_tokens": 2048, "packed": true},
  "layers": {
    "<layer_index>": {
      "total_tokens": 524288,
      "expert_frequency": [int, ...],
      "weighted_expert_frequency_sum": [float, ...],
      "ean_sum": [float, ...],
      "weighted_ean_sum": [float, ...],
      "ean_mean": [float, ...],
      "reap": [float, ...],
      "max_activations": [float, ...]
    }
  }
}
```

`observer-state.pt` is a torch state-dict you can `torch.load` to resume or
recompute derived metrics.

## Citation

REAP: Lasby et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE
Compression" 2025. The schema in this repo matches the Cerebras
`MoETransformerObserver` such that a downstream `reap-mlx`-style planner can
consume `expert-saliency.json` directly.

## License

MIT.
