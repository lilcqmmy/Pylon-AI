from datetime import datetime, timedelta, timezone

import pytest

from sfi.data import synthetic
from sfi.data.schema import Game

T0 = datetime(2024, 9, 8, 17, 0, tzinfo=timezone.utc)


def make_game(gid="G1", home="A", away="B", hs=24, as_=17, start=T0, season=2024, week=1, known=None):
    return Game(
        league="NFL", season=season, week=week, game_id=gid, home_team=home, away_team=away,
        scheduled_start_utc=start, home_score=hs, away_score=as_,
        result_known_at_utc=(known if known is not None else start + timedelta(hours=4)) if hs is not None else None,
        source_name="test",
    )


@pytest.fixture(scope="session")
def small_world():
    return synthetic.generate(seed=7, seasons=[2021, 2022, 2023], n_teams=12, weeks_per_season=10)
