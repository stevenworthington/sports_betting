from ..api import THIRTY_YEARS_DASH_STATS
from ..api.cache_filler import load_or_fetch_from_cache
from .feature_group import FeatureGroup
from .box_score_feature_group import FixedRollingWindow
from . import utility_functions as utl


class PrevMonthAdvancedStatsFeatureGroup(FeatureGroup):
    def __init__(self):
        self.df = None

    def process_data(self):
        if self.df is not None:
            return self.df
        self.df = load_or_fetch_from_cache(THIRTY_YEARS_DASH_STATS)
        drop_list = ["TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", "MONTH"]
        self.df = self.df.drop(columns=drop_list)
        return self.df

    def merge_data_to_base(self, base_df):
        base_df["MONTH"] = base_df.GAME_DATE.dt.month
        base_df["PREV_MONTH"] = base_df.MONTH.apply(lambda x: x - 1 if x > 1 else 12)
        merged = base_df
        for pref in ["HOME_", "AWAY_"]:
            merged = merged.merge(
                self.df,
                left_on=["SEASON_ID", f"{pref}TEAM_ID", "PREV_MONTH"],
                right_on=["SEASON", "TEAM_ID", "CALENDAR_MONTH"],
            )
            print(merged.columns)
            merged = merged.drop(columns=["SEASON", "CALENDAR_MONTH", "TEAM_ID"])
            merged = merged.rename(
                columns=lambda x: (f"{pref}{x}" if x in self.df.columns else x)
            )
            print(merged.columns)
        return merged
