from .base_builder import reshape_box_scores_to_matchups
from .feature_group import FeatureGroup
from .box_score_feature_group import FixedRollingWindow
from . import utility_functions as utl


class BoxScoreRatios(FeatureGroup):
    def __init__(self, box_score_feature_group=FixedRollingWindow()):
        self.box_score_feature_group = box_score_feature_group

    def process_data(self):
        df = self.box_score_feature_group.process_data().dropna()
        prefs = ["ROLL_HOME_", "ROLL_AWAY_"]
        cols = ["GAME_ID"]
        print(df.head())
        for pref in prefs:
            cur = f"{pref}3A_RATE"
            df[cur] = df[f"{pref}FG3A"] / df[f"{pref}FGA"]
            cols.append(cur)

            cur = f"{pref}ASSIST_TO_TURNOVER_RATE"
            df[cur] = df[f"{pref}AST"] / df[f"{pref}TOV"]
            cols.append(cur)

            cur = f"{pref}OTOD_REBRATE"
            df[cur] = df[f"{pref}OREB"] / df[f"{pref}DREB"]
            cols.append(cur)

        print(df.head())
        print(df.columns)
        self.df = df.reset_index()[cols].set_index("GAME_ID")
        return self.df

    def merge_data_to_base(self, base_df):
        return self.df.join(base_df.reset_index().set_index("GAME_ID"))
