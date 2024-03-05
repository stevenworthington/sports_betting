from ..james_utilities import load_and_scale_data
from .. import utilities as utl

from spam_data.feature_engineering import build_dataset

from sklearn.linear_model import LinearRegression


def train_first():
    df = build_dataset()
    pts_scaled_df, pm_scaled_df, res_scaled_df, test_set_obs = load_and_scale_data(
        df,
        seasons_to_keep=["2021-22", "2022-23", "2023-24"],
        training_season="2021-22",
        feature_prefix="ROLL_",
        scaler_type="minmax",
        scale_target=False,
    )

    season_22_ngames = 1186
    season_23_ngames = 1181

    # configuration for total points with rolling window
    model = LinearRegression()  # model class
    target_col = "TOTAL_PTS"  # target column name
    df = pts_scaled_df  # data set to use
    train_size = 1200  # size of the training set
    test_size = 1  # leave-one-out (LOO) cross-validation
    advancement_limit = None  # maximum number of times the training window is advanced

    # run model
    model_outputs, y_true = utl.train_with_rolling_window(
        model=model,
        target_col=target_col,
        df=df,
        train_size=train_size,
        test_size=test_size,
        advancement_limit=advancement_limit,
    )

    metrics = utl.calculate_metrics(y_true, model_outputs)
    print(metrics)


if __name__ == "__main__":
    train_first()
