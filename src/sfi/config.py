"""Load YAML configuration: config/config.yaml if present, else config/config.example.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        local = CONFIG_DIR / "config.yaml"
        path = local if local.exists() else CONFIG_DIR / "config.example.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("model", {}).get("llm_for_numeric_probability"):
        raise ValueError("Numeric probabilities must come from statistical models, not an LLM.")
    return cfg
