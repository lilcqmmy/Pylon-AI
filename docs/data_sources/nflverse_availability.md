# Data source evaluation: player availability (injuries, depth charts, QB starters)

Evaluated 2026-10-18. Goal: point-in-time player-availability features, starting with the QB, to close
part of the Brier gap to the betting market (≈0.0063–0.0070 on validation).

All outcome analysis below uses **validation seasons only (≤ 2019)**; the 2020–2025 holdout stays sealed.

## Sources

| Source | nflverse release | Seasons | Size | License |
|---|---|---|---|---|
| Injury reports (game status + practice participation) | `injuries` | 2009–2026 | 2.0 MB | CC-BY 4.0 |
| Depth charts | `depth_charts` | 2001–2026 | 16.1 MB | CC-BY 4.0 |
| Actual starting QB per game | `schedules` (`home_qb_id`, `away_qb_id`) | 1999–2026 | already held | CC-BY 4.0 |
| QB efficiency (`qb_epa`, `qb_dropback`, passer `id`) | `pbp` | 1999–2026 | already held | CC-BY 4.0 |

Downloaded with owner approval on 2026-10-18 (`sfi fetch --injuries 2009-2026 --depth-charts 2001-2026`),
stored as immutable checksummed per-season snapshots under git-ignored `data/raw/nflverse/`.

## Point-in-time findings (verified by profiling)

### Injury reports
- One row per player per team-week: `report_status` (Out / Doubtful / Questionable; "Probable" until
  2015; blank when only on the practice report), practice participation, injury, `date_modified`.
- **`date_modified` coverage:** 2009 essentially empty (17 of 4,821 rows); **2010–2024 complete**;
  **2025–2026 the column no longer exists.**
- Where present it is **before kickoff**: median ≈ 55 h before kickoff (2010–2020) and ≈ 47 h (2021–2024),
  5th percentile ≈ 29–40 h; only 0–13 rows per season were modified after kickoff. This matches the
  final (Friday) injury report. It records the *last* modification, so earlier versions and late
  game-day changes (e.g. Saturday downgrades, activations) are not visible.
- All team-weeks match the schedule except 2022 BUF/CIN week 17 (the cancelled game).
- Known-at policy for features: `date_modified` when present and before kickoff; rows modified after
  kickoff are dropped; when the column is missing (2025+), the report is treated as known at
  **kickoff − 24 h** (conservative versus the observed ≥ 29 h).

### Depth charts
- **Through 2024:** one chart per team-week (`season`, `week`, `depth_team`, `position`); no timestamp.
  QB1 matched the actual starter in ≈ 87–90% of team-games. On games where the starter changed from the
  previous game, QB1 more often still listed the **previous** starter (2010–2024, e.g. 35 vs 20 times in
  2024), so the charts are **not hindsight-filled** but lag real decisions. Before 2010 they matched the
  new starter slightly more often (2001: 22 vs 19; 2005: 32 vs 20): publication timing there is unclear.
- **From 2025:** daily snapshots with an ISO timestamp `dt` (median gap 24 h), genuinely point-in-time.
  QB1 at the last snapshot ≥ 1 h before kickoff matched the actual starter in 521 of 570 team-games (91%).

### Starting QB columns in schedules
`home_qb_id`/`away_qb_id` are the **actual** starters: hindsight, never a pregame feature. They are
used only as labels and for upper-bound analysis.

## How much signal? (validation 2001–2019, Elo = tuned Elo walk-forward)

- The starting QB changed from the team's previous game in **≈ 11% of team-games**.
- When only the home starter changed, the home team underperformed Elo by **−0.076** win probability on
  average (n = 481); when only the away starter changed, **+0.074** (n = 442).
- Logistic adjustments on top of the Elo logit, trained on earlier seasons and tested on later ones
  (paired Brier difference vs Elo, week-block bootstrap):

| Feature (train → test) | Pregame-available? | Test games | Brier diff vs Elo | 95% CI | On affected games |
|---|---|---:|---:|---|---:|
| QB-changed flags (2001–10 → 2011–19) | No (hindsight starter) | 2,395 | −0.0007 | [−0.0021, +0.0006] | −0.0031 |
| QB quality change: prior EPA/dropback, actual vs previous starter | No (hindsight starter) | 2,395 | −0.0008 | [−0.0015, −0.0001] | −0.0027 |
| Quality change + flags | No (hindsight starter) | 2,395 | **−0.0015** | [−0.0031, +0.0000] | −0.0069 |
| Previous starter Out/Doubtful/Questionable on injury report (2010–14 → 2015–19) | Yes | 1,330 | −0.0003 | [−0.0020, +0.0015] | −0.0035 |

(QB rating: cumulative `qb_epa` per dropback from earlier games, shrunk toward 0 with 300 dropbacks;
not tuned. These are exploratory fits; the train/test split is inside validation.)

**Reading:** QB availability carries real signal, concentrated in ~11% of games. But even with
*perfect knowledge* of who starts, the gain is about **0.0015 Brier overall**, roughly a quarter of the gap
to the market. A pregame-available version can only capture part of that, and the simple injury-report
proxy alone is not distinguishable from zero.

## Recommendation

Worth building, with modest expectations, as a **QB-availability adjustment** on top of Elo:

1. **Expected starter at as-of time:** the team's previous starter unless the latest point-in-time
   injury report lists him Out/Doubtful (then the next QB on the latest known depth chart). From 2025,
   use the daily depth-chart snapshot at or before as-of directly.
2. **QB rating:** point-in-time shrunk EPA/dropback from play-by-play of earlier games only.
3. **Feature:** expected starter's rating minus the team's recent-starter rating, home minus away, plus
   change flags; walk-forward ridge logistic like `pbp_logit`.
4. **Evaluation:** 2010–2019 (injury timestamps exist only from 2010), paired against `elo_tuned`;
   holdout stays sealed. Promote to live only with a clear validation gain.
5. **Live caveat:** 2025+ injury rows have no timestamp; the kickoff − 24 h policy must be documented in
   every live forecast's inputs.

Expected gain if it works: ≲ 0.0015 Brier. Remaining market gap after that would still be ≈ 0.005, likely
from information not in these feeds (late news, weather forecasts, line movement).
