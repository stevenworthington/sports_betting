from spam_ml.models.base_model import Model
from spam_ml import serialization as ser

from spam_data.feature_engineering import (
    build_dataset,
    FixedRollingWindow,
    BoxScoreRatios,
    PrevMonthAdvancedStatsFeatureGroup,
    LastMatchupFeatureGroup
)

import math
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class MlpRatiosAdvancedStats(Model):
    def __init__(self, name=None):
        super().__init__(name)

    def train(self):
        orig = build_dataset(
            [
                FixedRollingWindow(),
                BoxScoreRatios(),
                PrevMonthAdvancedStatsFeatureGroup(),
                LastMatchupFeatureGroup()
            ]
        )
        feature_cols = [
            "ROLL_HOME_PTS",
            "ROLL_AWAY_PTS",
            "ROLL_HOME_3A_RATE",
            "HOME_PACE",
            "HOME_OFF_RATING",
            "HOME_DEF_RATING",
            "ROLL_AWAY_3A_RATE",
            "AWAY_PACE",
            "AWAY_OFF_RATING",
            "AWAY_DEF_RATING",
            "LAST_MATCHUP_PTS",
            "LAST_MATCHUP_WINNER"
        ]
        target_col = ["TOTAL_PTS"]
        df = orig[feature_cols + target_col].dropna()
        scaler = StandardScaler()

        train_df, test_df = train_test_split(df, test_size=0.2)
        train_X = scaler.fit_transform(train_df.drop(columns=target_col).values)
        test_X = scaler.transform(test_df.drop(columns=target_col).values)

        self.model = MLPRegressor(hidden_layer_sizes=(20, 10, 5))
        self.model.fit(
            train_X,
            train_df[target_col].stack().values,
        )
        train_rmse = math.sqrt(
            mean_squared_error(
                test_df[target_col].stack().values,
                self.model.predict(test_X),
            )
        )
        print(f"{train_rmse=}")
        return train_df, test_df

    def save(self):
        super().save()
        import joblib

        model_path = os.path.join(self.path, "model.joblib")
        joblib.dump(self.model, model_path)
        print("Saving Complete")

    def load(self):
        super().load()
        import joblib

        model_path = os.path.join(self.path, "model.joblib")
        self.model = joblib.load(model_path)
        print("Load Successful")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="MlpRatiosAdvancedStats ", description="Simple Model Runner Template"
    )
    parser.add_argument("-n", "--name", help="Model name, if model is to be loaded")

    args = parser.parse_args()
    model = MlpRatiosAdvancedStats(name=args.name)
    if args.name is None:
        model.train()
        model.save()
        print(model.name)
