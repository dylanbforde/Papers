#!/usr/bin/env python3
"""One-shot JAX TPU health probe for Cloud TPU launchers."""

from __future__ import annotations

import os
import sys

import jax


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from jax_tpu_init import cloud_tpu_env_snapshot, maybe_initialize_jax_distributed


def _libtpu_version() -> str:
    try:
        import libtpu  # type: ignore
    except Exception:
        return "unavailable"
    return getattr(libtpu, "__version__", "unknown")


def main() -> int:
    maybe_initialize_jax_distributed(jax)

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported zero devices.")
    if devices[0].platform != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {devices[0].platform!r}.")

    result = jax.device_get(jax.numpy.add(1, 1))
    print("JAX TPU probe succeeded.")
    print(f"  jax={jax.__version__}")
    print(f"  jaxlib={jax.lib.__version__}")
    print(f"  libtpu={_libtpu_version()}")
    print(f"  process_index={jax.process_index()}")
    print(f"  process_count={jax.process_count()}")
    print(f"  device_count={jax.device_count()}")
    print(f"  local_device_count={jax.local_device_count()}")
    print(f"  sample_result={result}")
    print(f"  env={cloud_tpu_env_snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
