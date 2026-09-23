"""Public entrypoints for the Sinkhorn aligner implementation."""

from .ot_kernel import exact_sinkhorn_attention, sinkhorn_attention

__all__ = ["exact_sinkhorn_attention", "sinkhorn_attention"]
