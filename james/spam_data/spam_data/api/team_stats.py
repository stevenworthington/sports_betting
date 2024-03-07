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


def get_thirty_years_dash_team_stats():
    ldts_list = []
    seasons = []
    for year in range(1996, 2024):
        season = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season)
    months = range(1, 13)

    for season in seasons:
        for month in months:
            print(f"Querying season {season}, month {month}")
            # query for months
            ldts = leaguedashteamstats.LeagueDashTeamStats(month=month, season=season)
            # get the first DataFrame of those returned
            df = ldts.get_data_frames()[0]

            # add columns for 'season' and 'month'
            df["SEASON"] = season
            df["MONTH"] = month

            # append the DataFrame to the list
            ldts_list.append(df)

            # add time delay between requests
            time.sleep(3)

    # concatenate all DataFrames in the list into one large DataFrame
    ldts_df = pd.concat(ldts_list, ignore_index=True)
    return ldts_df


def get_advanced_team_stats_30_years():
    pass


if __name__ == "__main__":
    df = get_advanced_team_stats_by_month("november")
