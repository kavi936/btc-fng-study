"""Load study configuration from YAML and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root: btc_fng_study/
ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve absolute data/report paths.

    Environment variables from .env (if present) are loaded so
    COINGLASS_API_KEY is available without exporting manually.
    """
    load_dotenv(ROOT / ".env")
    path = config_path or (ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    for key in ("raw", "processed", "reports"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        cfg["paths"][key] = p
        p.mkdir(parents=True, exist_ok=True)

    cfg["coinglass_api_key"] = os.getenv("COINGLASS_API_KEY", "").strip() or None
    cfg["root"] = ROOT
    return cfg
