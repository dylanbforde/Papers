"""
Helpers for explicit JAX initialization on Cloud TPU.

The TPU launch scripts run on unstable, queued hardware where a silent fallback
to "let the first device query fail later" wastes hours. Keep initialization
centralized so benchmark entrypoints fail with actionable context instead.
"""

from __future__ import annotations

import os
import socket


_TRUE_VALUES = {"1", "true", "yes", "on"}
_TPU_ENV_KEYS = (
    "CLOUD_TPU_ACCELERATORS",
    "CLOUD_TPU_TASK_ID",
    "JAX_FORCE_TPU_INIT",
    "JAX_PLATFORMS",
    "LIBTPU_INIT_ARGS",
    "MEGATRON_WORKER",
    "PJRT_DEVICE",
    "PJRT_LOCAL_PROCESS_COUNT",
    "PJRT_LOCAL_PROCESS_RANK",
    "TPU_ACCELERATOR_TYPE",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_MESH_CONTROLLER_ADDRESS",
    "TPU_MESH_CONTROLLER_PORT",
    "TPU_PROCESS_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_TOPOLOGY_WRAP",
    "TPU_VISIBLE_CHIPS",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "XLA_FLAGS",
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def cloud_tpu_env_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in _TPU_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            snapshot[key] = value
    return snapshot


def cloud_tpu_runtime_snapshot() -> dict[str, str]:
    snapshot = cloud_tpu_env_snapshot()
    snapshot["hostname"] = socket.gethostname()
    return snapshot


def in_cloud_tpu_env() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "CLOUD_TPU_TASK_ID",
            "MEGATRON_WORKER",
            "TPU_ACCELERATOR_TYPE",
            "TPU_PROCESS_BOUNDS",
            "TPU_WORKER_HOSTNAMES",
            "TPU_WORKER_ID",
        )
    )


def should_initialize_jax_distributed() -> bool:
    if "MEGATRON_WORKER" in os.environ:
        return True
    return bool(os.environ.get("TPU_WORKER_HOSTNAMES"))


def maybe_initialize_jax_distributed(jax_module) -> bool:
    """Initialize JAX distributed mode when the TPU environment expects it."""
    if not should_initialize_jax_distributed():
        return False

    if getattr(maybe_initialize_jax_distributed, "_did_initialize", False):
        return True

    try:
        jax_module.distributed.initialize()
    except Exception as exc:  # pragma: no cover - exercised on TPU VMs
        env_summary = cloud_tpu_env_snapshot()
        raise RuntimeError(
            "JAX distributed initialization failed in the current TPU environment. "
            f"Relevant env: {env_summary}"
        ) from exc

    maybe_initialize_jax_distributed._did_initialize = True
    return True
