"""Generic REAP (Routed-Expert Activation Profiling) observer for HuggingFace MoE models.

REAP records, per MoE layer and per *routed* expert, four quantities:

    * expert_frequency[i]              — # of (token, top_k_slot) pairs that selected expert i
    * weighted_expert_frequency_sum[i] — Σ renormalized router weight contributed to expert i
    * ean_sum[i]                       — Σ ‖expert_i(x)‖₂ over tokens that chose expert i
    * weighted_ean_sum[i]              — Σ ‖expert_i(x)‖₂ × renorm_weight (the REAP signal)
    * max_activations[i]               — max |expert_i(x)| seen so far

From these the per-expert *means* are derived once at report time:

    ean_mean[i] = ean_sum[i]          / max(1, expert_frequency[i])
    reap[i]     = weighted_ean_sum[i] / max(1, expert_frequency[i])

The shared/always-on expert (if any) is structurally excluded from saliency — only experts
exposed under the configured ``experts_attr`` are observed.

----- Why this exists -----

The Cerebras REAP reference observer assumes the MoE block returns a 2-tuple
``(hidden_states, router_logits)`` with raw softmax-able logits. Several modern MoE
architectures (Nemotron-3 Nano, DeepSeek V3, Qwen3-MoE, etc.) instead return only the combined
hidden states from their forward, and apply sigmoid + grouped-topk + renormalization inside
the gate module — making the default observer crash or silently record nothing.

This file solves that with a small ``MoESpec`` dataclass that describes how to:
    * find MoE blocks   — by class name or substring match
    * call the gate     — gate(hidden_states) → either (topk_idx, topk_w) or router_logits
    * iterate experts   — attribute name of the routed-expert ``ModuleList``

Three built-in specs cover Nemotron-3 / DeepSeek-V3 / Qwen3-MoE; an ``auto`` resolver
heuristically detects most other HF MoE blocks. To add a new architecture, write a 4-line
``MoESpec`` and pass it in.

----- Bug fix vs. earlier private versions -----

Earlier private versions of this observer accumulated *per-batch means* into ``reap_sum`` and
``ean_mean_sum`` and divided by total ``expert_frequency`` at report time. That dampens
hot experts more than cold ones (because hot experts have higher per-batch routing density),
flattening the saliency landscape and distorting mid-band rankings. This file accumulates
*sums* and divides by frequency exactly once — see ``_apply_routing`` and ``report_state``.
"""
from __future__ import annotations

import gc
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoE block description
# ---------------------------------------------------------------------------
@dataclass
class MoESpec:
    """Describes how to observe one family of MoE blocks.

    name            : human-readable spec id
    block_classes   : exact class names (or substring matches) to hook on the model
    experts_attr    : attribute name of the routed-expert ``ModuleList`` (e.g. "experts")
    gate_attr       : attribute name of the router/gate module (e.g. "gate" or "router")
    gate_call       : "topk" if ``gate(x)`` returns ``(topk_idx, topk_w)``,
                      "logits" if ``gate(x)`` returns raw router_logits and we should
                      softmax + topk ourselves.
    layer_id_from_name : optional callable str -> int that extracts the layer index from
                         the qualified module name. Defaults to scanning for "layers.<i>".
    """

    name: str
    block_classes: tuple[str, ...]
    experts_attr: str = "experts"
    gate_attr: str = "gate"
    gate_call: str = "topk"  # "topk" or "logits"
    layer_id_from_name: Optional[Callable[[str], int]] = None

    def matches(self, module: nn.Module) -> bool:
        cn = module.__class__.__name__
        return any(cn == bc or bc in cn for bc in self.block_classes)


def _default_layer_id(name: str) -> int:
    """Extract layer index from names like ``backbone.layers.7.mixer`` or ``model.layers.3.mlp``."""
    for marker in ("layers.", "blocks.", "h."):
        if marker in name:
            try:
                return int(name.split(marker, 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
    raise ValueError(f"Could not extract layer index from module name: {name!r}")


# ---------------------------------------------------------------------------
# Built-in specs
# ---------------------------------------------------------------------------
SPEC_NEMOTRON_H = MoESpec(
    name="nemotron_h",
    block_classes=("NemotronHMOE",),
    experts_attr="experts",
    gate_attr="gate",
    gate_call="topk",  # gate returns (topk_idx, topk_w) already renormalized
)

SPEC_DEEPSEEK_V3 = MoESpec(
    name="deepseek_v3",
    block_classes=("DeepseekV3MoE", "DeepseekV2MoE"),
    experts_attr="experts",
    gate_attr="gate",
    gate_call="topk",
)

SPEC_QWEN3_MOE = MoESpec(
    name="qwen3_moe",
    block_classes=("Qwen3MoeSparseMoeBlock", "Qwen2MoeSparseMoeBlock"),
    experts_attr="experts",
    gate_attr="gate",
    gate_call="logits",  # gate returns raw router_logits; we softmax+topk
)

SPEC_MIXTRAL = MoESpec(
    name="mixtral",
    block_classes=("MixtralSparseMoeBlock",),
    experts_attr="experts",
    gate_attr="gate",
    gate_call="logits",
)

BUILTIN_SPECS: dict[str, MoESpec] = {
    s.name: s for s in (SPEC_NEMOTRON_H, SPEC_DEEPSEEK_V3, SPEC_QWEN3_MOE, SPEC_MIXTRAL)
}


def auto_detect_spec(model: nn.Module) -> MoESpec:
    """Best-effort heuristic: pick the first MoE-looking block class found in the model."""
    candidates: list[str] = []
    for _, module in model.named_modules():
        cn = module.__class__.__name__
        if "MoE" in cn or "Moe" in cn or "MOE" in cn or "SparseMoe" in cn:
            if hasattr(module, "experts") and isinstance(getattr(module, "experts"), nn.ModuleList):
                candidates.append(cn)
    if not candidates:
        raise RuntimeError(
            "auto-detect: no MoE-looking blocks found in model. "
            "Pass an explicit MoESpec via --moe-spec."
        )
    # Pick the most common class name (per-layer instances all share the same class)
    most = max(set(candidates), key=candidates.count)
    # Try built-ins first
    for spec in BUILTIN_SPECS.values():
        if any(bc == most or bc in most for bc in spec.block_classes):
            logger.info("auto-detect: matched built-in spec %r for class %r", spec.name, most)
            return spec
    # Fall back to a generic spec; assume gate is named "gate" and returns logits
    logger.warning(
        "auto-detect: no built-in spec for %r; using generic spec (gate_call='logits'). "
        "If this fails, define an explicit MoESpec.",
        most,
    )
    return MoESpec(name=f"auto:{most}", block_classes=(most,), gate_call="logits")


def resolve_spec(spec_id: str, model: nn.Module) -> MoESpec:
    if spec_id == "auto" or spec_id is None:
        return auto_detect_spec(model)
    if spec_id in BUILTIN_SPECS:
        return BUILTIN_SPECS[spec_id]
    raise ValueError(
        f"Unknown moe-spec {spec_id!r}. Choices: auto, {', '.join(BUILTIN_SPECS)}, "
        f"or import MoESpec and instantiate one."
    )


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
class MoEObserver:
    """Hook every MoE block matched by ``spec`` and accumulate REAP statistics on CPU."""

    def __init__(
        self,
        model: nn.Module,
        spec: MoESpec,
        num_routed_experts: int,
        top_k: int,
    ) -> None:
        self.model = model
        self.spec = spec
        self.num_routed_experts = int(num_routed_experts)
        self.top_k = int(top_k)
        self.state: dict[int, dict[str, Any]] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._hook_model()
        logger.info(
            "MoEObserver(%s) hooked %d MoE layers (E=%d, top_k=%d)",
            spec.name, len(self.hooks), self.num_routed_experts, self.top_k,
        )

    # -- setup -----------------------------------------------------------------

    def _hook_model(self) -> None:
        layer_id_fn = self.spec.layer_id_from_name or _default_layer_id
        seen = 0
        for name, module in self.model.named_modules():
            if not self.spec.matches(module):
                continue
            try:
                layer_idx = layer_id_fn(name)
            except Exception:
                layer_idx = seen
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(handle)
            seen += 1
        if not self.hooks:
            raise RuntimeError(
                f"MoEObserver({self.spec.name}): no modules matched "
                f"block_classes={self.spec.block_classes!r}. "
                "Check the spec or use --moe-spec auto."
            )

    def _initialize_layer_state(self, device: str = "cpu") -> dict[str, Any]:
        E = self.num_routed_experts
        return {
            "total_tokens": torch.zeros((), device=device, dtype=torch.long),
            "expert_frequency": torch.zeros(E, device=device, dtype=torch.long),
            "weighted_expert_frequency_sum": torch.zeros(E, device=device, dtype=torch.float64),
            "ean_sum": torch.zeros(E, device=device, dtype=torch.float64),
            "weighted_ean_sum": torch.zeros(E, device=device, dtype=torch.float64),
            "max_activations": torch.zeros(E, device=device, dtype=torch.float32),
        }

    # -- gate adapters ---------------------------------------------------------

    def _call_gate(self, module: nn.Module, hidden_states: torch.Tensor):
        gate = getattr(module, self.spec.gate_attr)
        out = gate(hidden_states)
        if self.spec.gate_call == "topk":
            # Expect (topk_idx, topk_w), in either order.
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError(
                    f"spec {self.spec.name}: gate_call='topk' but gate returned {type(out)}"
                )
            a, b = out[0], out[1]
            # The integer one is indices.
            if a.dtype in (torch.long, torch.int32, torch.int64):
                return a, b
            return b, a
        elif self.spec.gate_call == "logits":
            # Raw router logits → softmax + topk.
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits.float(), dim=-1)
            topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
            # Renormalize so chosen weights sum to 1 per token (Mixtral / Qwen3-MoE convention).
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return topk_idx, topk_w
        else:
            raise ValueError(f"spec {self.spec.name}: unknown gate_call {self.spec.gate_call!r}")

    # -- the hook itself -------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        E = self.num_routed_experts
        experts_attr = self.spec.experts_attr

        @torch.no_grad()
        def _hook_fn(module: nn.Module, args: tuple, output: Any) -> None:
            hidden_states = args[0]
            if hidden_states.dim() == 3:
                B, T, H = hidden_states.shape
                flat = hidden_states.reshape(-1, H)
            elif hidden_states.dim() == 2:
                flat = hidden_states
                H = flat.shape[1]
            else:
                raise ValueError(f"Unexpected hidden_states.dim()={hidden_states.dim()}")

            topk_idx, topk_w = self._call_gate(module, hidden_states)
            n_tok = topk_idx.shape[0]
            if topk_idx.dim() == 3:
                topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
                topk_w = topk_w.reshape(-1, topk_w.shape[-1])
                n_tok = topk_idx.shape[0]
            assert n_tok == flat.shape[0], (n_tok, flat.shape)

            if layer_idx not in self.state:
                self.state[layer_idx] = self._initialize_layer_state()
            ls = self.state[layer_idx]

            # Frequency counts.
            expert_frequency = torch.bincount(topk_idx.view(-1), minlength=E).to("cpu")
            ls["total_tokens"] += int(n_tok)
            ls["expert_frequency"] += expert_frequency

            weighted_freq_sum = torch.zeros(E, dtype=torch.float64)
            ean_sum = torch.zeros(E, dtype=torch.float64)
            weighted_ean_sum = torch.zeros(E, dtype=torch.float64)
            max_acts = ls["max_activations"].clone()

            tw_flat = topk_w.to(flat.device, flat.dtype)
            experts = getattr(module, experts_attr)

            for expert_idx in range(E):
                mask = (topk_idx == expert_idx)  # (n_tok, top_k)
                if not mask.any():
                    continue
                token_mask = mask.any(dim=-1)
                expert_input = flat[token_mask]
                expert_output = experts[expert_idx](expert_input)
                norms = torch.linalg.norm(expert_output.float(), dim=-1)
                weights_per_token = tw_flat[token_mask][mask[token_mask]].float()

                # SUMS, not means — the entire point of the bug-fix.
                ean_sum[expert_idx] = norms.sum().double().cpu()
                weighted_ean_sum[expert_idx] = (norms * weights_per_token).sum().double().cpu()
                weighted_freq_sum[expert_idx] = weights_per_token.sum().double().cpu()

                max_v = expert_output.abs().max().float().cpu()
                if max_v > max_acts[expert_idx]:
                    max_acts[expert_idx] = max_v

            ls["weighted_expert_frequency_sum"] += weighted_freq_sum
            ls["ean_sum"] += ean_sum
            ls["weighted_ean_sum"] += weighted_ean_sum
            ls["max_activations"] = max_acts

            del topk_idx, topk_w, tw_flat, expert_frequency
            gc.collect()

        return _hook_fn

    # -- public API ------------------------------------------------------------

    def report_state(self) -> dict[int, dict[str, Any]]:
        """Materialize the current accumulators as a JSON-friendly dict per MoE layer.

        ``reap`` and ``ean_mean`` are derived once here from the correctly-summed fields.
        """
        out: dict[int, dict[str, Any]] = {}
        for layer_idx, ls in self.state.items():
            ef = ls["expert_frequency"].clone()
            ef_safe = ef.clamp(min=1).to(torch.float64)
            tt = ls["total_tokens"].item() if isinstance(ls["total_tokens"], torch.Tensor) else int(ls["total_tokens"])
            wes = ls["weighted_ean_sum"]
            es = ls["ean_sum"]
            out[layer_idx] = {
                "total_tokens": int(tt),
                "expert_frequency": ef.tolist(),
                "weighted_expert_frequency_sum": ls["weighted_expert_frequency_sum"].tolist(),
                "ean_sum": es.tolist(),
                "weighted_ean_sum": wes.tolist(),
                "ean_mean": (es / ef_safe).tolist(),
                "reap": (wes / ef_safe).tolist(),
                "max_activations": ls["max_activations"].tolist(),
            }
        return out

    def save_state(self, path: pathlib.Path | str) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state, path)

    def reset(self) -> None:
        self.state = {}

    def close_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []
