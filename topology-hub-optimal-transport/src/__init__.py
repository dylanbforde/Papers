"""Topology Hub Optimal Transport (THOT): kernels, causal layer, and capacity gates."""

from .causal import CausalTHOTConfig, causal_thot_attention, causal_thot_head, verify_causal_invariance
from .gates import causal_threshold_gate, relu_threshold_capacities, sparsemax, sparsemax_capacities
from .kernels import (
    batched_logsumexp,
    batched_weighted_matvec,
    thot_attention,
    thot_reference,
    thot_transport,
)

__all__ = [
    "CausalTHOTConfig",
    "batched_logsumexp",
    "batched_weighted_matvec",
    "causal_thot_attention",
    "causal_thot_head",
    "causal_threshold_gate",
    "relu_threshold_capacities",
    "sparsemax",
    "sparsemax_capacities",
    "thot_attention",
    "thot_reference",
    "thot_transport",
    "verify_causal_invariance",
]

