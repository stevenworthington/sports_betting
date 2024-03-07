from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import (
    scoreboard,
    leaguegamefinder,
    leaguegamelog,
    playercareerstats,
)
from nba_api.stats.endpoints import (
    boxscorematchupsv3,
    boxscoreadvancedv2,
    boxscoreadvancedv3,
    teamestimatedmetrics,
    teamgamelogs,
    leaguedashteamstats,
)
from nba_api.stats.endpoints import TeamGameLogs, TeamEstimatedMetrics
import pandas as pd

from datetime import datetime, timedelta
import time
from functools import wraps
from functools import wraps


@wraps(leaguegamelog.LeagueGameLog)
def get_games(**kwargs):
    df = leaguegamelog.LeagueGameLog(**kwargs).get_data_frames()[0]
    aways = df[df["MATCHUP"].str.contains("@")]
    homes = df[~df["MATCHUP"].str.contains("@")]
    return aways.join(
        homes.set_index("GAME_ID"), on="GAME_ID", lsuffix="_AWAY", rsuffix="_HOME"
    )


def get_teams() -> pd.DataFrame:
    return pd.DataFrame(teams.get_teams())


def get_all_games_with_league_finder() -> pd.DataFrame:
    team_ids = get_teams()["id"].tolist()

    games_list = []

    for id in team_ids:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id)
        games_list.append(gamefinder.get_data_frames()[0])
        time.sleep(3)

    games_df = pd.concat(games_list)
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    return games_df


def get_games_within_seasons(start_season, end_season=22023):
    from . import cache_filler, ALL_GAMES_BOX_SCORES

    df = cache_filler.load_or_fetch_from_cache(ALL_GAMES_BOX_SCORES)
    r = list(range(start_season, end_season+1))
    return df[df.SEASON_ID.isin(r)]


def get_adv_stats_for_games_within_seasons(start_season, end_season=22023):
    game_df = get_games_within_seasons(start_season, end_season)
    print(game_df["GAME_ID"].tolist())
    return get_adv_stats_df(game_df["GAME_ID"].tolist())


def get_adv_stats_df(game_id_list):
    adv_games_stats_list = []
    for id in game_id_list:
        try:
            print(id)
            # query for games
            games = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=id)
            adv_games_stats_list.append(games.get_data_frames()[1])
            time.sleep(2)
        except IndexError:
            print(f"Error retrieving {id}")
            pass
    adv_stats_df = pd.concat(adv_games_stats_list, ignore_index=True)
    adv_stats_df = adv_stats_df.drop_duplicates()
    return adv_stats_df


if __name__ == "__main__":
    pass
