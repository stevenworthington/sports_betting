from nba_api.stats.endpoints import leaguegamelog
from functools import wraps


@wraps(leaguegamelog.LeagueGameLog)
def get_games(**kwargs):
    df = leaguegamelog.LeagueGameLog(**kwargs).get_data_frames()[0]
    aways = df[df["MATCHUP"].str.contains("@")]
    homes = df[~df["MATCHUP"].str.contains("@")]
    return aways.join(
        homes.set_index("GAME_ID"), on="GAME_ID", lsuffix="_AWAY", rsuffix="_HOME"
    )


if __name__ == "__main__":
    pass
