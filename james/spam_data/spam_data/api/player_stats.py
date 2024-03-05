from nba_api.stats.static import players
import pandas as pd


def get_players():
    return pd.DataFrame(players.get_players())
