from spam_ml.models.base_model import Model
from spam_ml import serialization as ser

from spam_data.feature_engineering import build_dataset, FixedRollingWindow
from spam_data.feature_engineering.box_score_ratios_feature_group import BoxScoreRatios

import math
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class JamesSimpleLinearRegressionWithRatios(Model):
    def __init__(self, name=None):
        super().__init__(name)

    def train(self):
        orig = build_dataset([FixedRollingWindow(), BoxScoreRatios()])
        season_id = orig.SEASON_ID.str.split("-").apply(lambda x: f"s{x[0]}")
        feature_cols = [
            "ROLL_HOME_PTS",
            "ROLL_AWAY_PTS",
            "ROLL_HOME_3A_RATE",
            "ROLL_HOME_ASSIST_TO_TURNOVER_RATE",
            "ROLL_HOME_OTOD_REBRATE",
            "ROLL_AWAY_3A_RATE",
            "ROLL_AWAY_ASSIST_TO_TURNOVER_RATE",
            "ROLL_AWAY_OTOD_REBRATE",
        ]
        target_col = ["TOTAL_PTS"]
        df = orig[feature_cols + target_col].dropna()
        df.join(pd.get_dummies(season_id))

        train_df, test_df = train_test_split(df, test_size=0.2)
        self.model = LinearRegression()
        self.model.fit(
            train_df.drop(columns=target_col).values,
            train_df[target_col].stack().values,
        )
        train_rmse = math.sqrt(
            mean_squared_error(
                test_df[target_col].stack().values,
                self.model.predict(test_df.drop(columns=target_col).values),
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
        prog="JamesSimpleLinearRegression ", description="Simple Model Runner Template"
    )
    parser.add_argument("-n", "--name", help="Model name, if model is to be loaded")

    args = parser.parse_args()
    model = JamesSimpleLinearRegressionWithRatios(name=args.name)
    if args.name is None:
        model.train()
        model.save()
        print(model.name)
