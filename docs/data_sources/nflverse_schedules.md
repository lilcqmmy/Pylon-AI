# Data source evaluation: nflverse schedules/results

Evaluated 2026-10-15. **Status: integrated** (`src/sfi/data/sources/nflverse.py`); owner confirmed
non-commercial research use with CC-BY attribution and the season splits below on 2026-10-15.

**Defect found during integration:** 101 night games in 2000–2005 have `gametime = "09:00"` instead of
`"21:00"` (all Mon/Thu/Sat; one more is the neutral-site 2003 MIA–SD game). Left as is, their results
would look known about 12 hours before kickoff. The adapter repairs exactly this pattern (season ≤ 2005
and `09:00`) and lists every repair; real 09:30 London kickoffs (2014+) are untouched.

Scope: game-level history (schedules, final scores, context) for the Elo/win-probability
baseline. Player-level feeds (play-by-play, injuries, snaps) are noted but not evaluated in depth here.

## Recommendation

Use **nflverse `schedules/games.csv`** as the first real data source, restricted to seasons **2000+**,
for **non-commercial research**. Do not use Sports Reference (Pro Football Reference) data or anything
scraped from it. Treat betting-line and weather columns as benchmark/diagnostic only, not model inputs.

## Candidates

| Source | License / terms | Verdict |
|---|---|---|
| **nflverse-data** (`schedules` release; data maintained in Lee Sharpe's `nfldata`) | Repository licensed **CC-BY 4.0**; nflreadpy README: most nflverse data "broadly licensed as CC-BY 4.0", FTN charting data **CC-BY-SA 4.0** | **Use** (attribution required) |
| Sports Reference / Pro Football Reference | Data use policy prohibits automated access without written permission, building competing data stores, and using data to train/prompt AI models | **Do not use** |
| nflverse snap counts / PFR advanced stats | nflverse documents these as sourced from PFR | **Avoid** until terms are clarified |
| FTN charting (via nflverse) | CC-BY-SA 4.0 (share-alike) | Later; share-alike obligations apply to derived data we publish |
| Commercial feeds (e.g. Sportradar, SportsDataIO) | Paid licenses | Not needed for MVP; revisit if live/real-time scale is required |
| Unofficial ESPN/NFL web endpoints | No published license | **Do not use** |

Caveats on nflverse's license: CC-BY 4.0 covers nflverse's compilation. Underlying facts (scores,
schedules) come from the league and are generally not copyrightable. The provenance and timing of the
betting-line columns are **undocumented**. Raw files stay out of git (`data/raw/` is ignored), so we
don't redistribute them.

Required attribution (README and any published output): *"Data: nflverse (nflverse-data, CC-BY 4.0);
schedules maintained by Lee Sharpe (nfldata)."*

## Snapshot inspected

- URL: `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv`
- Retrieved 2026-10-15T19:19:32Z; upstream `timestamp.json`: `2026-10-15 15:16:13 EDT`
- 2,184,451 bytes, SHA-256 `cf5b5f6bd3c3f4a1359e7c3e2bf47ab866b9a69789103e53a369d245d6638293`
- Stored at `data/raw/nflverse/schedules/20261015T191932Z/` (git-ignored)
- Upstream refresh cadence: every 5 minutes in season (nflverse automation schedule).

## What the data contains (verified by profiling)

- 7,548 games, seasons **1999–2026**, 46 columns; `game_id` unique (`{season}_{week}_{away}_{home}`).
- Game types: REG 7,239; playoffs WC 120, DIV 108, CON 54, SB 27.
- 103 **neutral-site** games (`location = Neutral`): Super Bowls and international games.
- 2026 season in progress: 79 of 272 games played (through 2026-10-12); no past-dated game lacks a score.
- Consistency checks passed: `result == home_score - away_score` for every played game; no team is
  booked twice in a (season, game_type, week).
- 35 team codes: relocations **OAK→LV (2020), SD→LAC (2017), STL→LA (2016)**; HOU from 2002.
- Ties: 0–2 per season. Overtime games: 6–26 per season.

Missing data (share of played games):

| Column | Gap |
|---|---|
| `gametime` | **100% missing in 1999**, complete 2000+ |
| `spread_line`, `total_line` | Complete 1999–2026 |
| moneylines / spread odds | Missing 1999–2005; partial 2006 (18%) and 2008 (27%); complete from 2010 |
| `temp`, `wind` | 20–62% missing (blank for domes / closed roofs by design, but also gaps outdoors) |
| `referee` | Complete except 18% in 2026 |
| `surface` | Up to 18% missing in recent seasons |
| `roof` | 31 games blank |

`gametime` is **Eastern time** regardless of venue (e.g. London games at `09:30`). Converting to UTC
needs a time-zone database; Windows Python has none, so the adapter needs the **`tzdata`** package
(verified: `ZoneInfo('America/New_York')` fails in our venv without it).

## Point-in-time (known-at) assessment

The file has **no per-row timestamps** for when a result or field became known, and rows are overwritten
in place (no revision history). So known-at times must come from explicit, recorded policies:

| Column group | Examples | Known when | Use |
|---|---|---|---|
| Identity / schedule | `game_id`, `season`, `game_type`, `week`, `gameday`, `gametime`, teams, `location`, `div_game`, `stadium_id`, `home_rest`/`away_rest` | When the schedule is published (months ahead; flexed games can move) | Features OK |
| Venue | `roof` dome/outdoors, `surface` | Pre-season | Features OK |
| Near-kickoff | retractable `roof` open/closed, starting QBs (`*_qb_id`), `referee` | Roughly game day / inactives (~90 min before kickoff) | **Not** for as-of earlier than kickoff−90 min; QB = actual starter (hindsight) |
| Outcome | `home_score`, `away_score`, `result`, `total`, `overtime` | After the game | Labels only |
| Market | `spread_line`, `total_line`, moneylines, odds | Undocumented (likely closing lines ≈ kickoff) | **Benchmark only**, never a feature |
| Game-time observation | `temp`, `wind` | During the game | Never a feature for pregame forecasts (a weather *forecast* source would be needed) |

Known-at policies for the adapter:

- `scheduled_start_utc` = `gameday` + `gametime` in America/New_York, converted to UTC.
- `result_known_at_utc` = kickoff + **5 hours** (`known_at_policy = "kickoff_plus_5h"`): games run about
  3h15m plus overtime and delays. Recorded per row, and set in config so it can be changed.
- Snapshot time = our `retrieved_at` plus upstream `timestamp.json`. Historical backtests use the final
  corrected data. Known limitation: score corrections are rare, and we can't see what was published at the time.
- Prospective (live) forecasts made from 2026 onward don't have this problem. Each forecast cites the
  snapshot it was made from.

## Adapter plan

### 1. Fetch (explicit command, network)
`sfi fetch nflverse-schedules`:
- Downloads `games.csv` and `timestamp.json` into `data/raw/nflverse/schedules/<retrieved_at_utc>/`.
  Never overwrites; a new folder for every pull.
- Writes `manifest.json`: source URL, retrieved_at, upstream last_updated, byte size, SHA-256, license
  `CC-BY-4.0`, attribution text, adapter version.
- Skips writing a new snapshot if the SHA-256 matches the latest one (reports "unchanged").

### 2. Normalize (offline, deterministic)
`src/sfi/data/sources/nflverse.py` → `load_snapshot(path) -> list[Game]`:
- Seasons **2000+** only (1999 has no kickoff times). Includes REG and playoff games.
- Extends `Game` (fields with defaults, so synthetic data keeps working): `game_type`, `neutral_site`,
  `franchise_home`/`franchise_away` (OAK/LV → `LV`, SD/LAC → `LAC`, STL/LA → `LA`), `source_record_id`,
  `known_at_policy`.
- Only schedule/identity/venue fields go into `Game`. Outcome fields fill `home_score`/`away_score`.
  Market and weather columns go to a separate benchmark table, so a model can't read them by accident.
- Unplayed games (NA scores) → `home_score=None`, `result_known_at_utc=None`.
- Validation: existing `validate_games` checks, plus `result`/`total` consistency, known team codes,
  kickoff ordering within a week, and one row per (season, game_type, week, team).

### 3. Model/harness changes needed for real data
- **Neutral site:** Elo home-field advantage set to 0 when `neutral_site`.
- **Franchise continuity:** Elo keyed by franchise, not team code.
- **Per-slate as-of:** forecast each kickoff window separately (Thu / Sun early / Sun late / SNF / MNF),
  so Thursday results inform Sunday forecasts. Today the whole week is forecast before Thursday.
- **Playoffs:** included; report REG and POST separately.
- **Market benchmark:** de-vigged moneyline implied probabilities (2010+) as a *comparison* row in reports,
  never as an input.

### 4. Proposed season splits (decide before looking at results)
- Burn-in 2000; validation 2001–2019; **holdout 2020–2025** (untouched until the model design is frozen).
  2020 had few or no fans, a useful stress test for home-field advantage.
- **2026 onward: prospective forecasts** archived before kickoff with `data_kind = "real"`. This is the
  cleanest out-of-sample test.

### 5. Tests
- Hand-written fixture rows in nflverse format (no copied data): ET→UTC across both DST changes,
  a London 09:30 ET game, neutral-site flag, relocation mapping, an unplayed game, a tie.
- Leakage guard: `Game` objects from the adapter expose no market/weather fields; known-at policy applied.
- Manifest/immutability: a second fetch with identical bytes creates no new snapshot.

### 6. Dependencies
- Add `tzdata` (needed on Windows). Keep the stdlib `csv` reader. Not adopting `nflreadpy` for now,
  since it pulls in Polars just to read one file; revisit when we need play-by-play.

## Open questions for the owner
1. Confirm non-commercial research use under CC-BY 4.0 with attribution (no redistribution of raw files).
2. Accept the proposed splits (holdout 2020–2025) before any real-data results are viewed?
3. Start prospective 2026 forecasts as soon as the adapter works (weeks 7–18 plus playoffs remain)?
