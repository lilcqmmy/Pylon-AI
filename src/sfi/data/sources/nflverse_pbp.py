"""nflverse play-by-play snapshots (one parquet file per season).

Data: nflverse (nflverse-data, CC-BY 4.0). Layout: `data/raw/nflverse/pbp/<season>/<retrieved_at>/`
with the raw `play_by_play_<season>.parquet` and our `manifest.json`. A season is re-downloaded only
when the upstream asset's size or update time changed, so refreshing during the season only pulls
the current season's file.

Caveat recorded in every manifest: nflverse EPA/success values come from the nflfastR expected-points
model, which was trained on many seasons; historical EPA therefore carries a small look-ahead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from sfi.data.sources.nflverse import ATTRIBUTION, LICENSE, _http_get, latest_snapshot

RELEASE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/pbp"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
ADAPTER_VERSION = "nflverse_pbp_v1"
EPA_CAVEAT = ("EPA/success come from the nflfastR expected-points model trained on many seasons; "
              "historical values carry a small look-ahead.")


def file_name(season: int) -> str:
    return f"play_by_play_{season}.parquet"


def _get_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "sfi-research"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        return json.loads(resp.read())


def list_assets(get_json: Callable[[str], dict] = _get_json) -> dict[str, dict]:
    release = get_json(RELEASE_API)
    return {a["name"]: {"size": a["size"], "updated_at": a["updated_at"]} for a in release["assets"]}


def fetch_pbp(
    root: Path, seasons: list[int], *, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    get: Callable[[str], bytes] = _http_get, get_json: Callable[[str], dict] = _get_json,
) -> list[tuple[int, Path, bool]]:
    """Returns [(season, snapshot folder, downloaded?)]. Unchanged upstream files are not downloaded."""
    assets = list_assets(get_json)
    out = []
    for season in seasons:
        name = file_name(season)
        if name not in assets:
            raise FileNotFoundError(f"{name} is not in the nflverse pbp release")
        meta = assets[name]
        season_root = root / str(season)
        latest = latest_snapshot(season_root)
        if latest is not None:
            m = json.loads((latest / "manifest.json").read_text(encoding="utf-8"))
            if m["upstream_asset_updated_at"] == meta["updated_at"] and m["bytes"] == meta["size"]:
                out.append((season, latest, False))
                continue
        data = get(BASE_URL + name)
        if len(data) != meta["size"]:
            raise ValueError(f"{name}: got {len(data)} bytes, release lists {meta['size']}")
        retrieved = now()
        folder = season_root / retrieved.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tmp = folder.with_name(folder.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            (tmp / name).write_bytes(data)
            manifest = {
                "source_name": "nflverse_pbp", "season": season, "source_url": BASE_URL + name,
                "adapter_version": ADAPTER_VERSION,
                "retrieved_at_utc": retrieved.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "upstream_asset_updated_at": meta["updated_at"], "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "license": LICENSE, "attribution": ATTRIBUTION,
                "caveat": EPA_CAVEAT,
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp.rename(folder)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        out.append((season, folder, True))
    return out


def latest_files(root: Path, seasons: list[int]) -> dict[int, tuple[Path, dict]]:
    """Latest snapshot file + manifest per season; raises if any season is missing or tampered."""
    out = {}
    for season in seasons:
        folder = latest_snapshot(root / str(season))
        if folder is None:
            raise FileNotFoundError(f"no play-by-play snapshot for {season}; run `sfi fetch --pbp {season}`")
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        path = folder / file_name(season)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != m["sha256"]:
            raise ValueError(f"{path} does not match its manifest checksum")
        out[season] = (path, m)
    return out


def combined_snapshot_id(manifests: list[dict]) -> str:
    blob = "|".join(f"{m['season']}:{m['sha256']}" for m in sorted(manifests, key=lambda m: m["season"]))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
