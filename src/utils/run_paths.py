from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str | None

    data_base: Path
    raw_cdx_dir: Path
    raw_html_dir: Path
    candidates_dir: Path
    extracted_text_dir: Path
    processed_dir: Path
    logs_dir: Path

    reports_dir: Path


def validate_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None

    run_id = run_id.strip()

    if not run_id:
        return None

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "run_id inválido. Usa solo letras, números, guion, guion bajo o punto. "
            f"Valor recibido: {run_id!r}"
        )

    return run_id


def get_run_paths(root: Path, run_id: str | None = None) -> RunPaths:
    run_id = validate_run_id(run_id)

    if run_id is None:
        data_base = root / "data"

        return RunPaths(
            root=root,
            run_id=None,
            data_base=data_base,
            raw_cdx_dir=data_base / "raw_cdx",
            raw_html_dir=data_base / "raw_html",
            candidates_dir=data_base / "candidates",
            extracted_text_dir=data_base / "extracted_text",
            processed_dir=data_base / "processed",
            logs_dir=data_base / "logs",
            reports_dir=root / "outputs" / "reports",
        )

    data_base = root / "data" / "runs" / run_id

    return RunPaths(
        root=root,
        run_id=run_id,
        data_base=data_base,
        raw_cdx_dir=data_base / "raw_cdx",
        raw_html_dir=data_base / "raw_html",
        candidates_dir=data_base / "candidates",
        extracted_text_dir=data_base / "extracted_text",
        processed_dir=data_base / "processed",
        logs_dir=data_base / "logs",
        reports_dir=root / "outputs" / "reports" / run_id,
    )