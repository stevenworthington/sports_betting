def load_and_scale_data(
    df,
    seasons_to_keep,
    training_season,
    feature_prefix,
    scaler_type="minmax",
    scale_target=False,
    csv_out=False,
    output_path=None,
):
    """
    Loads data from a specified file, filters for specific seasons, scales the features (and optionally the target),
    using only the training data for scaler fitting, and applies this scaling across specified seasons.

    Parameters:
    - file_path (str): The file path to the CSV containing the data.
    - seasons_to_keep (list): A list of SEASON_IDs to include in the analysis.
    - training_season (str): The season that the scaler should be fitted on.
    - scaler_type (str): The type of scaler to use for feature scaling ('minmax' or 'standard').
    - scale_target (bool): Whether to scale the target variable(s) alongside the features.
    - csv_out(bool): Whether to return csv files with the missing / non-missing observations meta-data
    - output_path (str, optional): Relative file path for the csv files.

    Returns:
    - Tuple of pd.DataFrame: Scaled DataFrames for features (and targets if `scale_target` is True) for each target variable.
      Specifically returns DataFrames for TOTAL_PTS, PLUS_MINUS, and GAME_RESULT targets.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # load the dataset
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # filter the DataFrame for the specified seasons
    df_filtered = df[df["SEASON_ID"].isin(seasons_to_keep)]

    # initialize the scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scaler must be either 'minmax' or 'standard'")

    # define feature names
    feature_names = [
        col for col in df_filtered.columns if col.startswith(feature_prefix)
    ]

    # identify rows with missing values in any of the feature_names and extract non-statistical data
    missing_features_rows = df_filtered[df_filtered[feature_names].isnull().any(axis=1)]
    dropped_obs_data = missing_features_rows[
        ["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME_TEAM_NAME", "AWAY_TEAM_NAME"]
    ]

    # identify rows with no missing values in any of the feature_names and extract non-statistical data
    no_missing_features_rows = df_filtered[
        ~df_filtered[feature_names].isnull().any(axis=1)
    ]
    kept_obs_data = no_missing_features_rows[
        ["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME_TEAM_NAME", "AWAY_TEAM_NAME"]
    ]

    # extract non-statistical data for non-missing rows in the test set (season 2023-24)
    kept_test_set_obs = kept_obs_data[kept_obs_data["SEASON_ID"] == "2023-24"]

    if csv_out and (output_path is not None):
        dropped_obs_data.to_csv(
            f"{output_path}nba_rolling_box_scores_dropped_observations.csv", index=False
        )
        kept_obs_data.to_csv(
            f"{output_path}nba_rolling_box_scores_kept_observations.csv", index=False
        )
        kept_test_set_obs.to_csv(
            f"{output_path}nba_rolling_box_scores_test_set_observations.csv",
            index=False,
        )

    # drop rows with missing values from df_filtered
    df_filtered = df_filtered.dropna(subset=feature_names)

    # set GAME_DATE as the index
    df_filtered.set_index("GAME_DATE", inplace=True)

    # fit the scaler on features from the training season only
    training_features = df_filtered[df_filtered["SEASON_ID"] == training_season][
        feature_names
    ]
    scaler.fit(training_features)

    # prepare to store scaled data for each target
    pts_scaled_df, pm_scaled_df, res_scaled_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    # apply scaling to features (and optionally targets) for all specified target columns
    for target_col in ["TOTAL_PTS", "PLUS_MINUS", "GAME_RESULT"]:

        # optionally scale target
        if scale_target:
            df_filtered[target_col] = scaler.transform(df_filtered[[target_col]])

        # select and assign scaled data for each target_col
        if target_col == "TOTAL_PTS":
            pts_scaled_df = df_filtered[feature_names + [target_col]].copy()
        elif target_col == "PLUS_MINUS":
            pm_scaled_df = df_filtered[feature_names + [target_col]].copy()
        elif target_col == "GAME_RESULT":
            res_scaled_df = df_filtered[feature_names + [target_col]].copy()

    # print the number of unique games for each season in the filtered data
    for season_id in seasons_to_keep:
        num_games = df_filtered[df_filtered["SEASON_ID"] == season_id][
            "GAME_ID"
        ].nunique()
        print(f"Season {season_id}: {num_games} games")

    # print total number of unique games in the filtered data
    print(
        f"Total number of games across sampled seasons: {df_filtered['GAME_ID'].nunique()} games"
    )

    return pts_scaled_df, pm_scaled_df, res_scaled_df, kept_test_set_obs
