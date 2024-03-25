from . import (
    game_stats,
    team_stats,
    ALL_GAMES_BOX_SCORES,
    RECENT_GAMES_BOX_SCORES,
    THIRTY_YEARS_DASH_STATS,
    BOX_SCORE_ADVANCED_STATISTICS,
)
from .. import SPAM_DATA_ROOT

import os

import pandas as pd


files = {
    ALL_GAMES_BOX_SCORES: game_stats.get_all_games_with_league_finder,
    RECENT_GAMES_BOX_SCORES: lambda: game_stats.get_games_within_seasons(22021),
    THIRTY_YEARS_DASH_STATS: team_stats.get_thirty_years_dash_team_stats,
    BOX_SCORE_ADVANCED_STATISTICS: lambda: game_stats.get_adv_stats_for_games_within_seasons(
        22021
    ),
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


def clear_cache(file_name: str):
    if file_name not in files:
        return False
    file_path = os.path.join(api_data_cache_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


if __name__ == "__main__":
    import argparse

    # TODO(JAMES)
