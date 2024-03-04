from .base_builder import reshape_box_scores_to_matchups
from .feature_group import FeatureGroup
from . import utility_functions as utl


class BoxScoreFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__()
        self.df = reshape_box_scores_to_matchups()
        self.processed = False


class FixedRollingWindow(BoxScoreFeatureGroup):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def process_data(self):
        if self.processed:
            return self.df

        non_stats_cols = [
            "SEASON_ID",
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "HOME_TEAM_NAME",
            "AWAY_TEAM_NAME",
            "HOME_WL",
            "AWAY_WL",
            "HOME_MIN",
            "AWAY_MIN",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
        ]
        stats_cols = [col for col in self.df.columns if col not in non_stats_cols]

        # calculate rolling averages for each statistic and add them to the DataFrame
        self.df = utl.process_rolling_stats(
            self.df,
            stats_cols,
            target_cols=["GAME_RESULT", "TOTAL_PTS", "PLUS_MINUS"],
            window_size=self.window_size,  # the number of games to include in the rolling window
            min_obs=1,  # the minimum number of observations present within the window to yield an aggregate value
            stratify_by_season=True,  # should the rolling calculations be reset at the start of each new season or be contiguous across seasons?
            exclude_initial_games=0,  # number of initial games to exclude from the rolling averages (optionally by season)
        )

        cols_to_drop = [
            "GAME_RESULT",
            "TOTAL_PTS",
            "PLUS_MINUS",
            "HOME_TEAM_NAME",
            "SEASON_ID",
            "GAME_DATE",
            "AWAY_TEAM_NAME",
        ]  # these are maintained by the base data

        self.df = self.df.drop(columns=cols_to_drop)
        self.df = self.df.set_index("GAME_ID")

        self.processed = True
        return self.df

    def merge_data_to_base(self, base_df):
        return self.df.join(base_df.set_index("GAME_ID"))
