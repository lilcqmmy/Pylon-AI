"""Named datasets (synthetic, nflverse) with their season splits and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sfi.config import PROJECT_ROOT
from sfi.data import synthetic
from sfi.data.schema import Game, file_sha256, read_games_csv
from sfi.data.sources import nflverse

SYN_DIR = PROJECT_ROOT / "data" / "synthetic"
SYN_GAMES = SYN_DIR / "games.csv"
SYN_TRUTH = SYN_DIR / "truth_DO_NOT_USE_AS_FEATURE.csv"


class DatasetError(RuntimeError):
    pass


def parse_seasons(spec) -> frozenset[int]:
    """Accepts 2000, "2000", "2001-2019", or a list mixing those."""
    if spec is None:
        return frozenset()
    if isinstance(spec, list):
        out: set[int] = set()
        for s in spec:
            out |= parse_seasons(s)
        return frozenset(out)
    if isinstance(spec, int):
        return frozenset({spec})
    s = str(spec).strip()
    if "-" in s:
        a, b = (int(x) for x in s.split("-", 1))
        if b < a:
            raise ValueError(f"bad season range {spec!r}")
        return frozenset(range(a, b + 1))
    return frozenset({int(s)})


@dataclass(frozen=True)
class Splits:
    burn_in: frozenset[int]
    validation: frozenset[int]
    holdout: frozenset[int]

    @classmethod
    def from_config(cls, d: dict) -> "Splits":
        sp = cls(parse_seasons(d.get("burn_in")), parse_seasons(d["validation"]), parse_seasons(d["holdout"]))
        sp.check()
        return sp

    def check(self, available: set[int] | None = None) -> None:
        if not self.validation or not self.holdout:
            raise DatasetError("validation and holdout seasons must be non-empty")
        if (self.burn_in & self.validation) or (self.burn_in & self.holdout) or (self.validation & self.holdout):
            raise DatasetError("burn-in, validation and holdout seasons must be disjoint")
        if (self.burn_in and max(self.burn_in) >= min(self.validation)) or max(self.validation) >= min(self.holdout):
            raise DatasetError("seasons must be ordered burn-in < validation < holdout")
        if available is not None:
            missing = (self.burn_in | self.validation | self.holdout) - available
            if missing:
                raise DatasetError(f"split seasons {sorted(missing)} are not in the data")


@dataclass
class Dataset:
    name: str
    data_kind: str  # "synthetic" | "real"
    games: list[Game]
    snapshot_id: str
    splits: Splits
    benchmarks: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)  # raw-snapshot manifest (real data only)
    lines: dict = field(default_factory=dict)     # market spread/total lines (benchmark only)

    def games_through(self, last_season: int) -> list[Game]:
        """Games up to and including `last_season`; later seasons are never loaded into a run."""
        return [g for g in self.games if g.season <= last_season]


def load_dataset(cfg: dict, name: str) -> Dataset:
    dcfg = cfg["datasets"].get(name)
    if dcfg is None:
        raise DatasetError(f"unknown dataset {name!r}; choose from {sorted(cfg['datasets'])}")
    splits = Splits.from_config(dcfg["splits"])
    if name == "synthetic":
        if not SYN_GAMES.exists():
            raise DatasetError("no synthetic data yet; run `sfi synth` first")
        games = read_games_csv(SYN_GAMES)
        truth = {gid: t.p_home_win_given_no_tie for gid, t in synthetic.read_truth_csv(SYN_TRUTH).items()}
        ds = Dataset(name, "synthetic", games, "sha256:" + file_sha256(SYN_GAMES)[:16], splits,
                     benchmarks={"oracle_true_prob": truth})
    elif name == "nflverse":
        root = PROJECT_ROOT / dcfg["raw_dir"]
        folder = nflverse.latest_snapshot(root)
        if folder is None:
            raise DatasetError(f"no nflverse snapshot under {dcfg['raw_dir']}; run `sfi fetch` first")
        snap = nflverse.load_snapshot(folder, min_season=dcfg["min_season"], known_at_hours=dcfg["known_at_hours"])
        m = snap.manifest
        ds = Dataset(
            name, "real", snap.games, snap.snapshot_id, splits,
            benchmarks={"market_moneyline": snap.market_p_home},
            notes=[f"Source: {m['source_url']} retrieved {m['retrieved_at_utc']} "
                   f"(upstream updated {m['upstream_last_updated']}), folder `{folder.name}`",
                   f"Known-at policy: result known {dcfg['known_at_hours']}h after kickoff; "
                   f"{len(snap.repairs)} documented kickoff-time repairs applied",
                   m["attribution"]],
            manifest=m, lines=snap.lines or {},
        )
    else:
        raise DatasetError(f"no loader for dataset {name!r}")
    splits.check({g.season for g in ds.games})
    return ds
