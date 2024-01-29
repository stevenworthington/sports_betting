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


def get_advanced_team_stats(**kwargs):
    return leaguedashteamstats.LeagueDashTeamStats(**kwargs).get_data_frames()[0]


if __name__ == "__main__":
    df = get_advanced_team_stats_by_month("november")
