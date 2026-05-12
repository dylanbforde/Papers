#!/usr/bin/env python3
# ruff: noqa: E402
"""One-shot JAX TPU health probe for Cloud TPU launchers."""

from __future__ import annotations

import os
import platform
import sys
import traceback

import jax


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from jax_tpu_init import cloud_tpu_runtime_snapshot, maybe_initialize_jax_distributed


def _libtpu_version() -> str:
    try:
        import libtpu  # type: ignore
    except Exception:
        return "unavailable"
    return getattr(libtpu, "__version__", "unknown")


def main() -> int:
    runtime = cloud_tpu_runtime_snapshot()
    print("JAX TPU probe starting.")
    print(f"  python={platform.python_version()}")
    print(f"  jax={jax.__version__}")
    print(f"  jaxlib={jax.lib.__version__}")
    print(f"  libtpu={_libtpu_version()}")
    print(f"  runtime={runtime}")
    try:
        maybe_initialize_jax_distributed(jax)

        devices = jax.devices()
        if not devices:
            raise RuntimeError("JAX reported zero devices.")
        if devices[0].platform != "tpu":
            raise RuntimeError(f"Expected TPU backend, got {devices[0].platform!r}.")

        result = jax.device_get(jax.numpy.add(1, 1))
        print("JAX TPU probe succeeded.")
        print(f"  process_index={jax.process_index()}")
        print(f"  process_count={jax.process_count()}")
        print(f"  device_count={jax.device_count()}")
        print(f"  local_device_count={jax.local_device_count()}")
        print(f"  devices={[str(device) for device in devices]}")
        print(f"  sample_result={result}")
        return 0
    except Exception as exc:
        print("JAX TPU probe failed.")
        print(f"  runtime={runtime}")
        print(f"  error={exc!r}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
