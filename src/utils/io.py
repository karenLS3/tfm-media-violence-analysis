from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(
    data: Any,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_json(
    path: str | Path,
) -> Any:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        path,
        index=False,
    )


def load_parquet(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)

    return pd.read_parquet(path)