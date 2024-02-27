from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import scoreboard, leaguegamefinder, playercareerstats
from nba_api.stats.endpoints import (
    boxscorematchupsv3,
    boxscoreadvancedv2,
    teamestimatedmetrics,
    teamgamelogs,
    leaguedashteamstats,
)
from nba_api.stats.endpoints import TeamGameLogs, TeamEstimatedMetrics
import pandas as pd

from datetime import datetime, timedelta
import time
from functools import wraps


month_map = {
    "november": 2,
    "december": 3,
    "january": 4,
    "february": 5,
    "march": 6,
    "april": 7,
    "may": 8,
}


def get_advanced_team_stats_by_month(month: str):
    return get_advanced_team_stats(month=month_map[month])


@wraps(leaguedashteamstats.LeagueDashTeamStats)
def get_advanced_team_stats(**kwargs):
    return leaguedashteamstats.LeagueDashTeamStats(**kwargs).get_data_frames()[0]


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


def get_advanced_team_stats_30_years():
    pass


if __name__ == "__main__":
    df = get_advanced_team_stats_by_month("november")
