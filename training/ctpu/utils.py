from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
# Required by strict CUDA deterministic algorithms for cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch

from .config import PROJECT_VERSION


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_int_seed(*parts: Any, bits: int = 63) -> int:
    s = "||".join(map(str, parts)).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=16).digest()
    return int.from_bytes(h, "little") & ((1 << bits) - 1)


def json_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def object_hash(obj: Any) -> str:
    return hashlib.sha256(json_canonical(obj).encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Flush the checkpoint contents before the atomic rename. This is especially
    # important on mounted cloud storage such as Google Drive.
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        # Avoid TF32-dependent drift across Ampere runtimes. BF16 autocast remains
        # enabled separately for transformer compute on supported GPUs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Hard fail rather than silently warning if an operation is nondeterministic.
        torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])




def reproducibility_environment_fingerprint() -> Dict[str, Any]:
    """Stable software/hardware fingerprint used to guard resume and final evaluation.

    Volatile fields such as total GPU memory, hostname and wall-clock time are
    excluded. CUDA compute capability is retained because a different GPU
    architecture can change numerical kernels even under deterministic mode.
    """
    import platform
    import sklearn
    import transformers
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    gpu_capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy_version,
        "gpu_compute_capability": gpu_capability,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }

def environment_report() -> Dict[str, Any]:
    import platform
    import sklearn
    import transformers

    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": props.total_memory / (1024**3),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "project_version": PROJECT_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy_version,
        "gpu": gpu,
        "determinism": {
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def choose_micro_batch(effective_batch_size: int = 32) -> tuple[int, int]:
    if not torch.cuda.is_available():
        micro = min(4, effective_batch_size)
    else:
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gb >= 60:
            micro = min(32, effective_batch_size)
        elif gb >= 34:
            micro = min(16, effective_batch_size)
        elif gb >= 20:
            micro = min(8, effective_batch_size)
        else:
            micro = min(4, effective_batch_size)
    while effective_batch_size % micro != 0 and micro > 1:
        micro //= 2
    accum = max(1, effective_batch_size // micro)
    return micro, accum


def normalize_text_for_hash(text: str) -> str:
    return " ".join(str(text).lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def flatten_text_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(flatten_text_field(v) for v in x if v is not None)
    if isinstance(x, dict):
        # Preserve insertion order from JSON.
        return " ".join(flatten_text_field(v) for v in x.values())
    return str(x)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_rmtree(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)



def tree_hash(path: str | Path) -> str:
    """Stable SHA256 over relative names + file SHA256 for a file tree."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    files = sorted(x for x in p.rglob("*") if x.is_file())
    payload = {str(x.relative_to(p)): sha256_file(x) for x in files}
    return object_hash(payload)
