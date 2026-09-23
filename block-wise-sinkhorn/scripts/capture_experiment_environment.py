#!/usr/bin/env python3
"""Emit a JSON environment snapshot for TPU/CPU experiment artifacts."""

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from jax_tpu_init import cloud_tpu_runtime_snapshot  # noqa: E402
except ImportError:
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

    def cloud_tpu_runtime_snapshot():
        snapshot = {}
        for key in _TPU_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                snapshot[key] = value
        snapshot["hostname"] = socket.gethostname()
        return snapshot


PACKAGES = (
    "jax",
    "jaxlib",
    "libtpu",
    "numpy",
    "optax",
    "scipy",
)


def _package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_dirty():
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--short"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _jax_snapshot():
    try:
        import jax
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "process_index": getattr(device, "process_index", None),
                "repr": str(device),
            }
        )
    return {
        "available": True,
        "default_backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "device_count": jax.device_count(),
        "devices": devices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    payload = {
        "run_label": args.run_label,
        "hostname": socket.gethostname(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "git": {
            "commit": _git_commit(),
            "dirty": _git_dirty(),
        },
        "jax": _jax_snapshot(),
        "tpu_environment": cloud_tpu_runtime_snapshot(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
