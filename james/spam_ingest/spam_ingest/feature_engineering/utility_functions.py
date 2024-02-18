# Utility functions for Capstone project


#################################################################################
##### Function to create a table of missing values
#################################################################################
def get_missing_values(df):
    """
    Parameters
    ----------
    df : a pandas DataFrame

    Returns:
    table : a displayed table of missing values
    """
    import numpy as np
    import pandas as pd

    missing_values = df.isnull().sum()
    missing_percentage = 100 * df.isnull().sum() / len(df)
    missing_values_table = pd.concat([missing_values, missing_percentage], axis=1)

    missing_values_table = missing_values_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )

    missing_values_table = (
        missing_values_table[missing_values_table.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )

    return missing_values_table


#################################################################################
##### Function to clean the team-level box score data from the nba_api
#################################################################################
def clean_team_bs_data(df, season_start_dates, season_end_dates, season_labels):
    """
    Clean the basketball data by filtering by season dates,
    replacing SEASON_ID with new labels, and printing the number of games per season.

    :param df: DataFrame containing team box score team data.
    :param season_start_dates: A list of starting dates for each season to filter the DataFrame. Dates should be 'YYYY-MM-DD'.
    :param season_end_dates: A list of ending dates for each season to filter the DataFrame. Dates should be 'YYYY-MM-DD'.
    :param season_labels: A list of labels for each season.
    :return: Cleaned DataFrame.
    """
    import pandas as pd

    # filter the number of minutes to keep only games with 238 or more
    df = df[df["MIN"] >= 238]

    # convert GAME_DATE to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # initialize an empty DataFrame for the cleaned data
    df_reg_seasons = pd.DataFrame()

    # map each game to its season and replace SEASON_ID
    for i, (start_date, end_date) in enumerate(
        zip(season_start_dates, season_end_dates)
    ):
        # Select games within the current season date range
        mask = (df["GAME_DATE"] >= start_date) & (df["GAME_DATE"] <= end_date)
        df_season = df.loc[mask].copy()
        if not df_season.empty:
            df_season["SEASON_ID"] = season_labels[i]
            df_reg_seasons = pd.concat([df_reg_seasons, df_season])

    # print the number of games in each season using the SEASON_ID labels
    for label in sorted(set(season_labels)):
        num_games = df_reg_seasons[df_reg_seasons["SEASON_ID"] == label][
            "GAME_ID"
        ].nunique()
        print(f"Season {label}: {num_games} games")

    return df_reg_seasons


#################################################################################
##### Function to reshape team box score data to wide format for matchups
#################################################################################
def reshape_team_bs_to_matchups(df, non_stats_cols):
    """
    Reshape the team box score data so that each game matchup is a single row with both home and away team data.

    :param df: DataFrame containing team box score data in long format.
    :param non_stats_cols: List of non-statistical column names that should not be renamed.
    :return: Reshaped DataFrame in wide format.
    """
    import pandas as pd

    # filter for home games and rename columns
    home_games_df = (
        df[df["MATCHUP"].str.contains(" vs. ")]
        .rename(
            columns={
                col: f"HOME_{col}" for col in df.columns if col not in non_stats_cols
            }
        )
        .drop("MATCHUP", axis=1)
    )

    # filter for away games and rename columns
    away_games_df = (
        df[df["MATCHUP"].str.contains(" @ ")]
        .rename(
            columns={
                col: f"AWAY_{col}" for col in df.columns if col not in non_stats_cols
            }
        )
        .drop("MATCHUP", axis=1)
    )

    # drop the non-stats columns from the away DataFrame
    away_games_df.drop(["SEASON_ID", "GAME_DATE"], axis=1, inplace=True)

    # merge home and away DataFrames on 'GAME_ID'
    wide_df = pd.merge(home_games_df, away_games_df, on="GAME_ID")

    season_game_counts = wide_df.groupby("SEASON_ID")["GAME_ID"].nunique()
    for season, count in season_game_counts.items():
        print(f"Season {season}: {count} games")

    return wide_df


#################################################################################
##### Function to calculate target variables for team-level data
#################################################################################
def create_target_variables(df, home_wl_col, home_pts_col, away_pts_col):
    """
    Calculate target variables for team matchups DataFrame.

    :param df: DataFrame containing team matchups data.
    :param home_wl_col: Column name for the home team win/loss status.
    :param home_pts_col: Column name for the home team points.
    :param away_pts_col: Column name for the away team points.
    :return: DataFrame with added target variable columns.
    """
    # calculate the game result for the home team
    df["GAME_RESULT"] = df[home_wl_col].apply(lambda x: 1 if x == "W" else 0)

    # calculate the total points scored
    df["TOTAL_PTS"] = df[home_pts_col] + df[away_pts_col]

    # calculate the score difference
    df["PLUS_MINUS"] = df[home_pts_col] - df[away_pts_col]

    return df


#################################################################################
##### Function to calculate rolling average statistics
#################################################################################
def calculate_rolling_stats(df, team_col, stats_cols, window_size, min_obs):
    """
    Calculate rolling statistics for a given team.

    :param df: DataFrame containing the team data
    :param team_col: Column name of the team in the DataFrame
    :param stats_cols: List of columns to include in rolling statistics
    :param window_size: Size of the rolling window
    :param min_obs: Minimum number of observations to calculate rolling stats
    :return: DataFrame with rolling statistics
    """
    # determine whether to use 'HOME' or 'AWAY' stats
    prefix = "HOME_" if "HOME" in team_col else "AWAY_"

    # filter the stats columns based on the prefix
    filtered_stats_cols = [col for col in stats_cols if col.startswith(prefix)]

    # ensure data is sorted by team, season, and date for accurate rolling calculation
    sorted_df = df.sort_values(by=[team_col, "SEASON_ID", "GAME_DATE"]).set_index(
        "GAME_ID"
    )

    # calculate rolling averages for each statistic
    rolling_stats = (
        sorted_df.groupby([team_col, "SEASON_ID"])[filtered_stats_cols]
        .rolling(window=window_size, min_periods=min_obs)
        .mean()
        .round(3)
        .shift(1)  # lag of 1 to exclude current game from the rolling average
        .add_prefix("ROLL_")
    )

    # reset the index for merging
    rolling_stats.reset_index(inplace=True)

    return rolling_stats


#################################################################################
##### Function to calculate rolling average statistics and add to a DataFrame
#################################################################################
def process_rolling_stats(df, stats_cols, window_size, min_obs):
    """
    Process the DataFrame to add rolling statistics for home and away teams, with rolling calculations
    resetting at the start of each new season.

    :param df: The original DataFrame.
    :param stats_cols: List of columns for rolling statistics.
    :param window_size: The size of the rolling window.
    :param min_obs: Minimum number of observations for rolling calculation.
    :return: DataFrame with added rolling statistics.
    """
    # calculate rolling stats for home and away teams
    rolling_home_stats = calculate_rolling_stats(
        df, "HOME_TEAM_NAME", stats_cols, window_size, min_obs
    )
    rolling_away_stats = calculate_rolling_stats(
        df, "AWAY_TEAM_NAME", stats_cols, window_size, min_obs
    )

    # merge the rolling stats into the original DataFrame
    final_df = df.merge(
        rolling_home_stats.drop("HOME_TEAM_NAME", axis=1),
        how="left",
        on=["GAME_ID", "SEASON_ID"],
    )
    final_df = final_df.merge(
        rolling_away_stats.drop("AWAY_TEAM_NAME", axis=1),
        how="left",
        on=["GAME_ID", "SEASON_ID"],
    )

    return final_df


#################################################################################
##### Function to plot targets against rolling average statistics over time
#################################################################################
def plot_team_bs_stats(df, team_col, feature_prefix, n_rows=3, n_cols=3):
    """
    Generate line plots for team statistics.

    :param df: DataFrame containing the team data.
    :param team_col: Column name indicating the team.
    :param feature_prefix: Prefix for the columns to plot.
    :param n_rows: Number of rows in the subplot grid.
    :param n_cols: Number of columns in the subplot grid.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    df["nth_game"] = df.groupby(team_col)["GAME_DATE"].rank(method="first").astype(int)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axs = axs.ravel()

    for i, col in enumerate(plot_cols):
        if i < len(axs):
            for team in df[team_col].unique():
                team_data = df[df[team_col] == team]
                axs[i].plot(
                    team_data["nth_game"], team_data[col], label=team, alpha=0.3
                )
            axs[i].set_title(col)
            axs[i].set_xlabel("n-th Game")
            axs[i].set_ylabel(col)

    for i in range(len(plot_cols), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.show()


#################################################################################
##### Function to scale data for modeling
#################################################################################
def scale_data(features, target, scaler="minmax", scale_target=False):
    """
    Scale the features (and optionally the target) of a dataset.

    Parameters:
    - features: DataFrame containing the features to scale.
    - target: Series or DataFrame containing the target variable.
    - scaler: String, either 'minmax' for MinMaxScaler or 'standard' for StandardScaler.
    - scale_target: Boolean, whether to scale the target variable or not.

    Returns:
    - DataFrame with scaled features (and target if scale_target is True), with index reset.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # initialize the scaler based on the input
    if scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scaler must be either 'minmax' or 'standard'")

    # scale features
    scaled_features = scaler.fit_transform(features)

    # prepare column names for the resulting DataFrame
    col_names = features.columns.tolist()

    if scale_target:
        # scale target if required using the same scaler initialized above
        scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
        # combine scaled features and target
        scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)
        # add the target column name
        col_names.append(target.name)
    else:
        # combine scaled features with unscaled target by ensuring target is properly shaped
        target_array = (
            target.values.reshape(-1, 1) if len(target.shape) == 1 else target
        )
        scaled_data = np.concatenate([scaled_features, target_array], axis=1)
        # add the target column name
        col_names.append(target.name)

    # convert scaled data back to DataFrame, assign column names, and reset index
    scaled_data_df = pd.DataFrame(scaled_data, columns=col_names)
    scaled_data_df = scaled_data_df.reset_index(drop=True)

    return scaled_data_df


#################################################################################
##### Function to load, filter (by time) and scale data for modeling
#################################################################################
def load_and_scale_data(
    file_path, seasons_to_keep, scaler_type="minmax", scale_target=False
):
    """
    Loads data from a specified file, filters for specific seasons, scales the features,
    and returns three DataFrames each targeted to a different outcome variable.

    Parameters:
    - file_path (str): The file path to the CSV containing the data.
    - seasons_to_keep (list): A list of SEASON_IDs to include in the analysis.
    - scaler_type (str): The type of scaler to use for feature scaling. Defaults to 'minmax'.
                         Acceptable values are 'minmax' for MinMaxScaler and 'standard' for StandardScaler.
    - scale_target (bool): Whether to scale the target variable or just the features.

    Returns:
    - Tuple of DataFrames: (pts_scaled_df, pm_scaled_df, res_scaled_df) where each DataFrame
      corresponds to scaled features with one of the targets being TOTAL_PTS, PLUS_MINUS, and GAME_RESULT.
    """
    import pandas as pd

    # load the dataset
    df = pd.read_csv(file_path)

    # convert 'GAME_DATE' column to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # filter the DataFrame for the specified seasons
    df_filtered = df[df["SEASON_ID"].isin(seasons_to_keep)]

    # print the number of unique games for each season in the filtered data
    for season_id in seasons_to_keep:
        num_games = df_filtered[df_filtered["SEASON_ID"] == season_id][
            "GAME_ID"
        ].nunique()
        print(f"Season {season_id}: {num_games} games")

    # feature columns
    feature_names = [col for col in df_filtered.columns if col.startswith("ROLL_")]
    features = df_filtered[feature_names]

    # target columns
    target_pts = df_filtered["TOTAL_PTS"]
    target_pm = df_filtered["PLUS_MINUS"]
    target_res = df_filtered["GAME_RESULT"]

    # scale data using a helper function 'scale_data' that you would need to define
    pts_scaled_df = scale_data(
        features, target_pts, scaler=scaler_type, scale_target=scale_target
    )
    pm_scaled_df = scale_data(
        features, target_pm, scaler=scaler_type, scale_target=scale_target
    )
    res_scaled_df = scale_data(
        features, target_res, scaler=scaler_type, scale_target=scale_target
    )

    # print total number of unique games in the filtered data
    print(
        f"Total number of games across sampled seasons: {df_filtered['GAME_ID'].nunique()} games"
    )

    return pts_scaled_df, pm_scaled_df, res_scaled_df


#################################################################################
##### Function to create an expanding window time series split
#################################################################################
def expanding_window_ts_split(
    df,
    initial_train_size,
    test_size=1,
    ensure_diversity=False,
    target_col=None,
    expansion_limit=None,
):
    """
    Generate indices to split data into training and test sets for expanding window time series cross-validation,
    with an option to ensure that the initial training set includes a diversity of classes.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be split, which should contain both features and the target column.
    - initial_train_size (int): The initial size of the training set, which will expand in each subsequent split.
    - test_size (int): The size of the test set for each split. Default is 1.
    - ensure_diversity (bool): If True, adjusts the initial_train_size to ensure the training set starts with both classes present, if the target column's diversity allows for it. Default is False.
    - target_col (str): The name of the target column to check for class diversity. Required if ensure_diversity is True.
    - expansion_limit (int, optional): The maximum number of times the training set is expanded by 1 observation during the expanding window process. This parameter controls the total number of train-test splits generated, indirectly determining the final size of the training set. If set, the training process will stop once this limit is reached, potentially leaving some data unused. If None, the training set will expand until all but the last observation are used for training.

    Yields:
    - train_indices (np.array): Indices for the training set for each split, which expands in size over iterations.
    - test_indices (np.array): Indices for the test set for each split.

    Raises:
    - ValueError: If the dataset is not large enough for the specified initial train size and test size, or if the initial training set does not include both classes when ensure_diversity is True.
    """
    import numpy as np

    if ensure_diversity and target_col is not None:
        target_values = df[target_col].values
        second_class_start = np.min(np.where(target_values != target_values[0])[0])
        initial_train_size = max(initial_train_size, second_class_start + 1)

    if len(df) < initial_train_size + test_size:
        raise ValueError(
            "Dataset is not large enough for the specified initial train size and test size."
        )

    indices = np.arange(len(df))
    expansion_count = 0  # initialize expansion count

    for start_index in range(initial_train_size, len(df) - test_size + 1):
        if expansion_limit is not None and expansion_count >= expansion_limit:
            break  # stop yielding new splits once the expansion limit is reached

        train_indices = indices[:start_index]
        test_indices = indices[start_index : start_index + test_size]
        yield train_indices, test_indices

        expansion_count += 1  # increment expansion count for each iteration


#################################################################################
##### Function to train models on an expanding window time series split
#################################################################################
def train_with_expanding_window(
    df,
    initial_train_size,
    test_size,
    target_col,
    model,
    ensure_diversity=False,
    expansion_limit=None,
):
    """
    Trains a given model using an expanding window approach on a specified DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and target variable.
    - initial_train_size (int): The initial size of the training dataset.
    - test_size (int): The size of the test dataset for each split, typically 1 for LOO CV.
    - target_col (str): The name of the target column in `df`.
    - model (model object): The instantiated model to be trained, e.g., LinearRegression() or LogisticRegression().
    - ensure_diversity (bool, optional): For logistic regression, ensures the initial training data includes both classes. Default is False.
    - expansion_limit (int, optional): The maximum number of times the training set is expanded by 1 observation during the expanding window process. This parameter controls the total number of train-test splits generated, indirectly determining the final size of the training set. If set, the training process will stop once this limit is reached, potentially leaving some data unused. If None, the training set will expand until all but the last observation are used for training.

    Returns:
    - model_outputs (list): A list of model predictions or probabilities for the test sets across all splits.
    - y_true (list): A list of the actual target values corresponding to each prediction in `model_outputs`.

    This function iterates over the dataset using an expanding window to create training and test splits,
    trains the specified `model` on each training split, and stores the model's predictions or probabilities.
    """
    import time
    from xgboost import XGBClassifier, XGBRegressor

    start_time = time.time()

    # initialize storage for model outputs and true labels
    model_outputs = []  # store predictions or probabilities
    y_true = []

    for train_indices, test_indices in expanding_window_ts_split(
        df,
        initial_train_size,
        test_size=test_size,
        ensure_diversity=ensure_diversity,
        target_col=target_col if ensure_diversity else None,
        expansion_limit=expansion_limit,
    ):

        # get training and testing data for this window
        X_train = df.iloc[train_indices].drop(columns=target_col)
        y_train = df.iloc[train_indices][target_col]
        X_test = df.iloc[test_indices].drop(columns=target_col)
        y_test = df.iloc[test_indices][target_col]

        # train the model
        model.fit(X_train, y_train)

        # check if the model has the predict_proba method (i.e., likely a classifier)
        if hasattr(model, "predict_proba"):
            # store predicted probabilities of the positive class
            proba = model.predict_proba(X_test)[:, 1]
            model_outputs.extend(proba)
        elif hasattr(model, "predict"):
            # for models that support predict (regressors and classifiers without predict_proba)
            predictions = model.predict(X_test)
            model_outputs.extend(predictions)
        else:
            raise ValueError("Model does not support required prediction methods.")

        # store true labels for evaluation
        y_true.extend(y_test)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return model_outputs, y_true


#################################################################################
##### Function to create a rolling window time series split
#################################################################################
def rolling_window_ts_split(
    df,
    train_size,
    test_size,
    ensure_diversity=False,
    target_col=None,
    advancement_limit=None,
):
    """
    Generate indices to split data into training and test sets for rolling window time series cross-validation.

    Optionally ensures that the initial training set includes a diversity of classes for logistic regression training.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing both features and the target column.
    - train_size (int): The size of the training set for each split.
    - test_size (int): The size of the test set for each split.
    - ensure_diversity (bool): If True, checks to ensure the initial training set includes at least two classes. Default is False.
    - target_col (str): The name of the target column to check for class diversity. Required if ensure_diversity is True.
    - - advancement_limit (int, optional): Specifies the maximum number of times the training window is allowed to advance during the rolling window process. This parameter effectively limits the number of train-test splits generated, allowing for control over the number of models trained and tested, which can be useful for large datasets or for limiting computational expense. If not set, or if set to None, the window will advance until it reaches the end of the dataset, making use of all possible train-test splits given the `train_size` and `test_size` parameters. Setting this parameter helps in focusing the training and testing process on a specific subset of the dataset, potentially leaving some data unused at the end of the dataset.

    Yields:
    - train_indices (np.array): Indices for the training set for each split.
    - test_indices (np.array): Indices for the test set for each split.

    Raises:
    - ValueError: If the dataset is not large enough for the specified train and test sizes, or if ensure_diversity is True but the initial training set does not include both classes.
    """
    import numpy as np

    if ensure_diversity and target_col is not None:
        # ensure initial training set includes both classes, adjust train_size if necessary
        target_values = df[target_col].values
        unique_classes = np.unique(target_values[:train_size])
        if len(unique_classes) < 2:
            raise ValueError("Initial training set does not include both classes.")

    if len(df) < train_size + test_size:
        raise ValueError(
            "The dataset is not large enough for the specified train and test sizes."
        )

    indices = np.arange(len(df))
    max_start_index = len(df) - train_size - test_size + 1
    actual_advancements = (
        min(advancement_limit, max_start_index)
        if advancement_limit is not None
        else max_start_index
    )

    for start_index in range(actual_advancements):
        yield indices[start_index : start_index + train_size], indices[
            start_index + train_size : start_index + train_size + test_size
        ]


#################################################################################
##### Function to train models on an rolling window time series split
#################################################################################
def train_with_rolling_window(
    df,
    train_size,
    test_size,
    target_col,
    model,
    ensure_diversity=False,
    advancement_limit=None,
):
    """
    Trains a specified model using a rolling window approach on a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing both features and the target variable.
    - train_size (int): The size of the training set for each split.
    - test_size (int): The size of the test set for each split, typically 1 for leave-one-out cross-validation.
    - target_col (str): The name of the target column in `df`.
    - model (model object): The instantiated model to be trained. This can be any model that conforms to the scikit-learn model interface, such as instances of `LinearRegression` or `LogisticRegression`.
    - ensure_diversity (bool, optional): Indicates whether to ensure the initial training data includes a diverse set of classes for classification tasks. This is primarily relevant for logistic regression and similar models where class diversity in the training set might impact model training. Default is False.
    - advancement_limit (int, optional): Specifies the maximum number of times the training window is allowed to advance during the rolling window process. This parameter effectively limits the number of train-test splits generated, allowing for control over the number of models trained and tested, which can be useful for large datasets or for limiting computational expense. If not set, or if set to None, the window will advance until it reaches the end of the dataset, making use of all possible train-test splits given the `train_size` and `test_size` parameters. Setting this parameter helps in focusing the training and testing process on a specific subset of the dataset, potentially leaving some data unused at the end of the dataset.

    Returns:
    - model_outputs (list): A list of model predictions or probabilities for the test sets across all splits. For logistic regression models, this will be the probabilities of the positive class. For linear regression models, it will be direct predictions.
    - y_true (list): A list of actual target values corresponding to each prediction in `model_outputs`.

    The function iterates over the dataset using a rolling window to create training and test splits. It then trains the specified `model` on each training split and stores the model's predictions or probabilities for further evaluation.
    """
    import time

    start_time = time.time()

    # initialize storage for model outputs and true labels
    model_outputs = []  # store predictions or probabilities
    y_true = []

    # use the rolling window index function for data splits
    for train_indices, test_indices in rolling_window_ts_split(
        df,
        train_size,
        test_size,
        ensure_diversity=ensure_diversity,
        target_col=target_col if ensure_diversity else None,
        advancement_limit=advancement_limit,
    ):

        # get training and testing data for this window
        X_train = df.iloc[train_indices].drop(columns=target_col)
        y_train = df.iloc[train_indices][target_col]
        X_test = df.iloc[test_indices].drop(columns=target_col)
        y_test = df.iloc[test_indices][target_col]

        # train the model
        model.fit(X_train, y_train)

        # check if the model has the predict_proba method (i.e., likely a classifier)
        if hasattr(model, "predict_proba"):
            # store predicted probabilities of the positive class
            proba = model.predict_proba(X_test)[:, 1]
            model_outputs.extend(proba)
        elif hasattr(model, "predict"):
            # for models that support predict (regressors and classifiers without predict_proba)
            predictions = model.predict(X_test)
            model_outputs.extend(predictions)
        else:
            raise ValueError("Model does not support required prediction methods.")

        # store true labels for evaluation
        y_true.extend(y_test)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return model_outputs, y_true


#################################################################################
##### Function to calculate performance metrics from trained models
#################################################################################
def calculate_metrics(y_true, model_outputs, threshold=0.5, verbose=True):
    """
    Calculates and returns evaluation metrics based on the provided data, automatically determining
    whether to treat the task as a classification or regression problem by inspecting `y_true`.
    For classification tasks, a custom threshold can be specified for converting probabilities to binary labels.

    Parameters:
    - y_true (list or np.array): The actual target values.
    - model_outputs (list or np.array): The model's predictions or probabilities for classifiers, or direct predictions for regressors.
    - threshold (float, optional): The threshold for converting probabilities to binary labels in classification tasks. Default is 0.5.
    - verbose (bool, optional): Controls whether to print the calculated metrics. Default is True.

    For binary classification tasks, it calculates and conditionally prints Average Accuracy, Overall AUC, and Average F1 Score based on the verbose parameter.
    For regression tasks, it calculates and conditionally prints the Root Mean Squared Error (RMSE) based on the verbose parameter.

    Returns:
    - metrics (dict): A dictionary containing the calculated metrics.
    """
    from sklearn.metrics import (
        mean_squared_error,
        roc_auc_score,
        accuracy_score,
        f1_score,
    )
    import numpy as np

    metrics = {}  # dictionary to store calculated metrics

    unique_values = np.unique(y_true)
    if len(unique_values) == 2:  # binary classification
        pred_labels = [1 if p > threshold else 0 for p in model_outputs]
        metrics["pred_labels"] = pred_labels
        metrics["average_accuracy"] = accuracy_score(y_true, pred_labels)
        metrics["overall_auc"] = roc_auc_score(y_true, model_outputs)
        metrics["average_f1_score"] = f1_score(y_true, pred_labels)
        if verbose:
            print(
                f"Classification Metrics:\n- Average Accuracy: {metrics['average_accuracy']:.2f}\n- Overall AUC: {metrics['overall_auc']:.2f}\n- Average F1 Score: {metrics['average_f1_score']:.2f}"
            )
    else:  # regression
        metrics["average_rmse"] = mean_squared_error(
            y_true, model_outputs, squared=False
        )
        if verbose:
            print(f"Regression Metrics:\n- Average RMSE: {metrics['average_rmse']:.2f}")

    return metrics


#################################################################################
##### Function to calculate performance metrics from trained models
#################################################################################
def compile_results_to_dataframe(
    results, calculate_metrics_function=calculate_metrics, verbose=False
):
    """
    Compiles the results of model runs with various hyperparameters into a pandas DataFrame.

    This function iterates through each model run's results, calculates performance metrics using
    a specified metrics calculation function, and combines these metrics with the hyperparameters
    used for the run into a single DataFrame for easy analysis and comparison.

    Parameters:
    - results (dict): A dictionary containing the results of multiple model runs, where each key is a
                      run identifier and each value is another dictionary with keys 'params' (the
                      hyperparameters used), 'model_outputs' (the predictions made by the model), and
                      'y_true' (the actual target values).
    - calculate_metrics_function (function): A function that calculates the desired metrics from
                                             'model_outputs' and 'y_true'. This function should return
                                             a dictionary where each key is the name of a metric and
                                             each value is the metric's value. Default is 'calculate_metrics'.

    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a single model run, including columns
                    for each hyperparameter and each calculated metric. There is also a 'run_id' column
                    that uniquely identifies each run.
    """
    import pandas as pd

    data_for_df = []

    # iterate through each run in the results
    for run_id, run_data in results.items():
        # extract parameters and results for this run
        params = run_data["params"]
        y_true = run_data["y_true"]
        model_outputs = run_data["model_outputs"]

        # calculate metrics for this run
        metrics = calculate_metrics_function(y_true, model_outputs, verbose=verbose)

        # combine params and metrics into a single dictionary
        combined_data = {**params, **metrics}

        # add unique identifier for the run
        combined_data["run_id"] = run_id

        # append the combined data to our list
        data_for_df.append(combined_data)

    # create a DataFrame from the list of combined data
    metrics_df = pd.DataFrame(data_for_df)

    # reorder the DataFrame columns
    columns_order = ["run_id"] + sorted(
        [col for col in metrics_df.columns if col != "run_id"]
    )
    metrics_df = metrics_df[columns_order]

    return metrics_df


#################################################################################
##### Function to get best hyperparameter settings from validation set
#################################################################################
def get_best_params(df, metric):
    """
    Extracts the best hyperparameters based on a specified performance metric.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing model runs, their performance metrics, and hyperparameters.
    - metric (str): The name of the column representing the performance metric (e.g., 'average_rmse', 'accuracy').
                    The function will find the minimum value of this metric to determine the best parameters.

    Returns:
    - dict: A dictionary of the best hyperparameters, excluding the run ID and the metric itself.
    """
    # exclude 'run_id' and any metrics from the parameters
    params_to_exclude = [
        "run_id",
        "average_rmse",
        "average_accuracy",
        "average_f1_score",
        "overall_auc",
        "pred_labels",
    ]

    # determine whether to find the min or max value based on the metric
    if metric == "average_rmse":
        best_row_idx = df[metric].idxmin()
    elif metric in ["average_accuracy", "overall_auc", "average_f1_score"]:
        best_row_idx = df[metric].idxmax()

    # find the row with the optimal performance metric
    best_row = df.loc[best_row_idx]

    # create the dictionary of best parameters, excluding specified columns
    best_params = {
        param: best_row[param]
        for param in best_row.index
        if param not in params_to_exclude
    }

    return best_params


#################################################################################
##### Function to handle types that are not natively serializable in json files
#################################################################################
def handle_non_serializable(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))
