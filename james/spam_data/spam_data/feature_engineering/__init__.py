from .base_builder import create_base_with_targets
from .box_score_feature_group import FixedRollingWindow
from .box_score_ratios_feature_group import BoxScoreRatios
from .prev_month_advanced_stats_feature_group import PrevMonthAdvancedStatsFeatureGroup
from .last_matchup_feature_group import LastMatchupFeatureGroup


def build_dataset(feature_groups=[FixedRollingWindow()]):
    df = create_base_with_targets()
    for fg in feature_groups:
        df = fg(df)
    return df.reset_index().dropna()
