"""nflverse per-season release files (play-by-play, injuries, depth charts) as immutable snapshots.

Data: nflverse (nflverse-data, CC-BY 4.0). Layout: `<root>/<season>/<retrieved_at>/` with the raw
`<prefix>_<season>.parquet` and our `manifest.json`. A season is re-downloaded only when the upstream
asset's size or update time changed, so in-season refreshes only pull the current season's file.

Play-by-play caveat (recorded in its manifests): nflverse EPA/success values come from the nflfastR
expected-points model, which was trained on many seasons; historical EPA carries a small look-ahead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from sfi.data.sources.nflverse import ATTRIBUTION, LICENSE, _http_get, latest_snapshot

RELEASES_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/"
DOWNLOAD_BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
EPA_CAVEAT = ("EPA/success come from the nflfastR expected-points model trained on many seasons; "
              "historical values carry a small look-ahead.")


@dataclass(frozen=True)
class ReleaseSpec:
    tag: str
    prefix: str
    caveat: str = ""

    def file_name(self, season: int) -> str:
        return f"{self.prefix}_{season}.parquet"

    @property
    def adapter_version(self) -> str:
        return f"nflverse_{self.tag}_v1"


PBP = ReleaseSpec("pbp", "play_by_play", EPA_CAVEAT)
INJURIES = ReleaseSpec("injuries", "injuries",
                       "Only time field is date_modified (last update), not the report's publication time.")
DEPTH_CHARTS = ReleaseSpec("depth_charts", "depth_charts",
                           "Source changed after 2024: weekly charts before, timestamped (dt) snapshots from 2025.")
SPECS = {s.tag: s for s in (PBP, INJURIES, DEPTH_CHARTS)}


def file_name(season: int) -> str:  # play-by-play (kept for existing callers)
    return PBP.file_name(season)


def _get_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "sfi-research"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        return json.loads(resp.read())


def list_assets(spec: ReleaseSpec = PBP, get_json: Callable[[str], dict] = _get_json) -> dict[str, dict]:
    release = get_json(RELEASES_API + spec.tag)
    return {a["name"]: {"size": a["size"], "updated_at": a["updated_at"]} for a in release["assets"]}


def fetch_release(
    spec: ReleaseSpec, root: Path, seasons: list[int], *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    get: Callable[[str], bytes] = _http_get, get_json: Callable[[str], dict] = _get_json,
) -> list[tuple[int, Path, bool]]:
    """Returns [(season, snapshot folder, downloaded?)]. Unchanged upstream files are not downloaded."""
    assets = list_assets(spec, get_json)
    out = []
    for season in seasons:
        name = spec.file_name(season)
        if name not in assets:
            raise FileNotFoundError(f"{name} is not in the nflverse {spec.tag} release")
        meta = assets[name]
        season_root = root / str(season)
        latest = latest_snapshot(season_root)
        if latest is not None:
            m = json.loads((latest / "manifest.json").read_text(encoding="utf-8"))
            if m["upstream_asset_updated_at"] == meta["updated_at"] and m["bytes"] == meta["size"]:
                out.append((season, latest, False))
                continue
        url = DOWNLOAD_BASE + spec.tag + "/" + name
        data = get(url)
        if len(data) != meta["size"]:
            raise ValueError(f"{name}: got {len(data)} bytes, release lists {meta['size']}")
        retrieved = now()
        folder = season_root / retrieved.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tmp = folder.with_name(folder.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=False)
        try:
            (tmp / name).write_bytes(data)
            manifest = {
                "source_name": f"nflverse_{spec.tag}", "release": spec.tag, "season": season,
                "file_name": name, "source_url": url, "adapter_version": spec.adapter_version,
                "retrieved_at_utc": retrieved.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "upstream_asset_updated_at": meta["updated_at"], "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "license": LICENSE, "attribution": ATTRIBUTION,
                "caveat": spec.caveat,
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp.rename(folder)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        out.append((season, folder, True))
    return out


def latest_release_files(spec: ReleaseSpec, root: Path, seasons: list[int]) -> dict[int, tuple[Path, dict]]:
    """Latest snapshot file + manifest per season; raises if any season is missing or tampered."""
    out = {}
    for season in seasons:
        folder = latest_snapshot(root / str(season))
        if folder is None:
            raise FileNotFoundError(f"no {spec.tag} snapshot for {season}; run `sfi fetch --{spec.tag.replace('_', '-')} {season}`")
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        path = folder / spec.file_name(season)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != m["sha256"]:
            raise ValueError(f"{path} does not match its manifest checksum")
        out[season] = (path, m)
    return out


def fetch_pbp(root: Path, seasons: list[int], **kw) -> list[tuple[int, Path, bool]]:
    return fetch_release(PBP, root, seasons, **kw)


def latest_files(root: Path, seasons: list[int]) -> dict[int, tuple[Path, dict]]:
    return latest_release_files(PBP, root, seasons)


def combined_snapshot_id(manifests: list[dict]) -> str:
    blob = "|".join(f"{m['season']}:{m['sha256']}" for m in sorted(manifests, key=lambda m: m["season"]))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]
