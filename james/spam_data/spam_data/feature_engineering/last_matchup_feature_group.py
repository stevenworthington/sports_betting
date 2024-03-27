from .base_builder import reshape_box_scores_to_matchups
from .feature_group import FeatureGroup
from . import utility_functions as utl


class LastMatchupFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__()
        self.df = reshape_box_scores_to_matchups()
        self.processed = False

    def process_data(self):
        if self.processed:
            return self.df

        import numpy as np

        def last_matchup_pts(row):
            teams = [row.HOME_TEAM_ID, row.AWAY_TEAM_ID]
            r = self.df[
                (
                    (self.df.HOME_TEAM_ID.isin(teams))
                    & (self.df.AWAY_TEAM_ID.isin(teams))
                    & (self.df.GAME_DATE < row.GAME_DATE)
                )
            ]
            r = r[r.GAME_DATE == r.GAME_DATE.max()]
            return r.TOTAL_PTS.item() if not r.empty else np.nan

        def last_matchup_winner(row):
            teams = [row.HOME_TEAM_ID, row.AWAY_TEAM_ID]
            r = self.df[
                (
                    (self.df.HOME_TEAM_ID.isin(teams))
                    & (self.df.AWAY_TEAM_ID.isin(teams))
                    & (self.df.GAME_DATE < row.GAME_DATE)
                )
            ]
            r = r[r.GAME_DATE == r.GAME_DATE.max()]
            if r.empty:
                return np.nan
            winner = (
                r.HOME_TEAM_ID.item()
                if r.GAME_RESULT.item() == 1
                else r.AWAY_TEAM_ID.item()
            )
            return 1 if row.HOME_TEAM_ID == winner else 0

        self.df["LAST_MATCHUP_PTS"] = self.df.apply(last_matchup_pts, axis=1)
        self.df["LAST_MATCHUP_WINNER"] = self.df.apply(last_matchup_winner, axis=1)
        self.df = self.df.dropna()
        self.df["LAST_MATCHUP_PTS"] = self.df["LAST_MATCHUP_PTS"].astype(int)
        self.df["LAST_MATCHUP_WINNER"] = self.df["LAST_MATCHUP_WINNER"].astype(int)

        cols_to_keep = ["GAME_ID", "LAST_MATCHUP_PTS", "LAST_MATCHUP_WINNER"]
        self.df = self.df[cols_to_keep]
        self.df = self.df.set_index("GAME_ID")

        self.processed = True
        return self.df

    def merge_data_to_base(self, base_df):
        print(self.df.head(), base_df.columns)
        return self.df.join(base_df.reset_index().set_index("GAME_ID"))
