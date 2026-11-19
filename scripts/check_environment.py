from __future__ import annotations

import importlib
import platform
from pathlib import Path


def version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return f"MISSING ({type(exc).__name__})"


print(f"Python platform: {platform.platform()}")
for package in [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "torch",
    "transformers",
    "sentence_transformers",
    "datasets",
    "faiss",
    "onnxruntime",
]:
    print(f"{package:>24}: {version(package)}")

try:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
except Exception:
    pass

for path in [
    "data/raw/shopping_queries_dataset_examples.parquet",
    "data/raw/shopping_queries_dataset_products.parquet",
]:
    file_path = Path(path)
    print(f"{path}: {'OK' if file_path.exists() else 'MISSING'}")
