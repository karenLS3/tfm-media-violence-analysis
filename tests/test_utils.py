from pathlib import Path
import pandas as pd

from src.utils.io import (
    save_json,
    load_json,
    save_parquet,
    load_parquet,
)


def test_json_io(tmp_path: Path):
    data = {
        "a": 1,
        "b": "texto"
    }

    file_path = tmp_path / "test.json"

    save_json(data, file_path)

    loaded = load_json(file_path)

    assert loaded == data


def test_parquet_io(tmp_path: Path):
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"]
    })

    file_path = tmp_path / "test.parquet"

    save_parquet(df, file_path)

    loaded = load_parquet(file_path)

    assert loaded.equals(df)