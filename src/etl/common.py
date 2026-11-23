"""Shared utilities for ETL pipelines: config loading, HTTP retries, and LMIC metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
ASSUMPTIONS_PATH = PROJECT_ROOT / "config" / "business_assumptions.yaml"
USER_AGENT = "ai-labor-shock-observatory/1.0 (academic-research)"


def load_pipeline_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_business_assumptions(path: Path = ASSUMPTIONS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(
    url: str,
    dest: Path,
    timeout: int = 120,
    retries: int = 3,
    force: bool = False,
) -> int:
    """Stream a remote file to disk with exponential backoff retries."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                size_bytes = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
                            size_bytes += len(chunk)
            tmp_path.replace(dest)
            return size_bytes
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2.0 * attempt)

    assert last_exc is not None
    raise last_exc


def fetch_json(url: str, timeout: int = 45, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_exc is not None
    raise last_exc


def get_lmics_metadata(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Return canonical World Bank LMICs table (iso3, iso2, country_name, income_level, region)."""
    wb_cfg = cfg["sources"]["world_bank_wdi"]
    countries_path = PROJECT_ROOT / wb_cfg["local_countries_file"]
    if not countries_path.exists() or force:
        payload = fetch_json(wb_cfg["countries_lmics_url"], timeout=30)
        countries_path.parent.mkdir(parents=True, exist_ok=True)
        countries_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        payload = json.loads(countries_path.read_text(encoding="utf-8"))

    raw_countries = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    rows = []
    for c in raw_countries:
        region_id = (c.get("region") or {}).get("id", "")
        iso3 = (c.get("id") or "").strip().upper()
        if region_id == "NA" or not iso3:
            continue
        rows.append(
            {
                "iso3": iso3,
                "iso2": (c.get("iso2Code") or "").strip().upper(),
                "country_name": (c.get("name") or "").strip(),
                "income_level": ((c.get("incomeLevel") or {}).get("id") or "").strip(),
                "income_name": ((c.get("incomeLevel") or {}).get("value") or "").strip(),
                "region": ((c.get("region") or {}).get("value") or "").strip(),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["iso3"]).reset_index(drop=True)
