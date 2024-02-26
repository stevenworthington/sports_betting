from . import game_stats, team_stats
from .. import SPAM_DATA_ROOT

import os

import pandas as pd


files = {
    "nba_games_box_scores_1984_2024.csv": team_stats.get_all_games_with_league_finder
}


def load_or_fetch_from_cache(file_name: str):
    if file_name not in files:
        return None

    file_path = os.path.join(SPAM_DATA_ROOT, file_name)
    df = load_path_from_cache_or_none(file_path)
    if df is not None:
        return df
    df = files[file_name]()
    df.to_csv(file_path, index=False)
    return df


def load_from_cache_or_return_none(file_name: str):
    if file_name not in files:
        return None

    file_path = os.path.join(SPAM_DATA_ROOT, file_name)
    return load_path_from_cache_or_none(file_path)


def load_path_from_cache_or_none(file_path: str):
    if os.path.exists():
        return pd.read_csv(file_path)
    return None
