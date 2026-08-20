from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


def collect_metadata(
    *,
    experiment_name: str,
    dataset_path: str,
    dataset_sha256: str,
    features: list[str],
    target: str,
    seed: int,
    split_definition: dict,
    model_parameters: dict,
) -> dict:

    return {
        "experiment": experiment_name,
        "dataset": Path(dataset_path).name,
        "dataset_sha256": dataset_sha256,
        "features": features,
        "target": target,
        "seed": seed,
        "split": split_definition,
        "model_parameters": model_parameters,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }


def save_metadata(metadata: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )