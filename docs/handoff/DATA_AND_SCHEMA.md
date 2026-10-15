# Data contract and illustrative schemas

## Ingestion domains
Schedules, game results, play-by-play, team strength, players and roster membership, snaps/routes/targets/carries, injury/practice availability, depth chart, venues, weather, travel/rest, coaching/scheme changes, source corrections. Each feed needs documented coverage, licensing/permission and update frequency.

## Critical identifiers
Canonical league, season, week/round, game_id, team_id, player_id, source_name, source_record_id. Handle trades and roster changes with time validity. Preserve event time and `known_at`/`ingested_at` separately.

## Forecast JSON example — illustrative schema only
```json
{
  "forecast_id": "fcst_sample_001",
  "league": "NFL",
  "game_id": "sample_game_001",
  "outcome_type": "home_win",
  "prediction_as_of_utc": "2026-10-15T17:00:00Z",
  "scheduled_start_utc": "2026-10-18T17:00:00Z",
  "source_snapshot_id": "snapshot_example",
  "feature_spec_version": "features_v0",
  "model_name": "baseline_elo",
  "model_version": "0.1.0-example",
  "distribution": {"p_home_win": 0.64, "p_away_win": 0.36},
  "calibration_version": "example_only",
  "quality_gate": {"status": "publish", "reasons": []},
  "simulation_only": true
}
```
This is *not* a real event or a live prediction.

## Database tables (suggested)
- sources: provider, license, fields, refresh rules, reliability notes.
- raw_snapshot: snapshot_id, retrieved_at, content_hash, source, storage URI.
- event: IDs, start, venue, season, result state.
- roster_membership and player_status: effective time + known_at.
- event_stat and play_by_play: versioned revisions and provenance.
- feature_snapshot: as_of, feature version, missingness, content hash.
- model_registry: artifact location, train_end, evaluation IDs, approval.
- forecast: immutable run metadata, targets, distributions, intervals, publication state.
- actual_result: finalized outcome and correction version.
- forecast_score: metric result, evaluation version, scoring date.
- data_quality_incident: issues and resolution.

## Data QA gates
Schema validation; timestamp ordering; duplicate keys; player/team mapping; completeness; freshness; official result reconciliation; impossible stat combinations; outlier flagging, not blind deletion; provider license compliance.
