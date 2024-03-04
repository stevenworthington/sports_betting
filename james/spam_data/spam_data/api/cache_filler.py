from . import (
    game_stats,
    team_stats,
    ALL_GAMES_BOX_SCORES,
    RECENT_GAMES_BOX_SCORES,
    THIRTY_YEARS_DASH_STATS,
)
from .. import SPAM_DATA_ROOT

import os

import pandas as pd


files = {
    ALL_GAMES_BOX_SCORES: team_stats.get_all_games_with_league_finder,
    RECENT_GAMES_BOX_SCORES: lambda: team_stats.get_games_within_seasons(22021),
    THIRTY_YEARS_DASH_STATS: team_stats.get_thirty_years_dash_team_stats,
}


api_data_prefix_dir = "original"
api_data_cache_path = os.path.join(SPAM_DATA_ROOT, api_data_prefix_dir)
os.makedirs(api_data_cache_path, exist_ok=True)


def load_or_fetch_from_cache(file_name: str):
    if file_name not in files:
        return None

    file_path = os.path.join(api_data_cache_path, file_name)
    df = load_path_from_cache_or_none(file_path)
    if df is not None:
        return df
    df = files[file_name]()
    df.to_csv(file_path, index=False)
    return df


def load_from_cache_or_return_none(file_name: str):
    if file_name not in files:
        return None

    file_path = os.path.join(api_data_cache_path, file_name)
    return load_path_from_cache_or_none(file_path)


def load_path_from_cache_or_none(file_path: str):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None


if __name__ == "__main__":
    import argparse

    # TODO(JAMES)
