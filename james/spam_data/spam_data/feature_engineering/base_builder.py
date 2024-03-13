from ..api import cache_filler, RECENT_GAMES_BOX_SCORES
from . import utility_functions as utl


def reshape_box_scores_to_matchups():
    team_bs_df = cache_filler.load_or_fetch_from_cache(RECENT_GAMES_BOX_SCORES)
    season_start_dates = ["2021-10-19", "2022-10-18", "2023-10-24"]
    season_end_dates = ["2022-04-10", "2023-04-09", "2024-04-14"]
    season_labels = ["2021-22", "2022-23", "2023-24"]
    team_bs_df_cleaned = utl.clean_team_bs_data(
        team_bs_df,
        season_start_dates=season_start_dates,
        season_end_dates=season_end_dates,
        season_labels=season_labels,
    )
    # identify non-stats columns
    non_stats_cols = ["SEASON_ID", "GAME_ID", "GAME_DATE", "MATCHUP"]

    # reshape team box score data to wide format so each row is a game matchup
    team_bs_matchups_df = utl.reshape_team_bs_to_matchups(
        team_bs_df_cleaned, non_stats_cols
    )
    team_bs_matchups_df = utl.create_target_variables(
        team_bs_matchups_df, "HOME_WL", "HOME_PTS", "AWAY_PTS"
    )
    return team_bs_matchups_df


def create_base_with_targets():
    df = reshape_box_scores_to_matchups()
    base_columns = [
        "SEASON_ID",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "AWAY_TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "HOME_PTS",
        "AWAY_PTS",
        "GAME_RESULT",
        "TOTAL_PTS",
        "PLUS_MINUS",
    ]
    return df[base_columns]
