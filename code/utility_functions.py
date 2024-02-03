
# Utility functions for Capstone project

#################################################################################
##### Function to create a table of missing values
#################################################################################
def get_missing_values(df):
    '''
    Parameters
    ----------
    df : a pandas DataFrame
    
    Returns:
    table : a displayed table of missing values
    '''
    import numpy as np
    import pandas as pd
    
    missing_values = df.isnull().sum()
    missing_percentage = (100 * df.isnull().sum() / len(df))
    missing_values_table = pd.concat([missing_values, missing_percentage], axis=1)
    
    missing_values_table = missing_values_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    missing_values_table = missing_values_table[
    missing_values_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    return missing_values_table
    

#################################################################################
##### Function to clean the team-level box score data from the nba_api
#################################################################################
def clean_team_bs_data(df, game_id_col='GAME_ID', team_abbrev_col='TEAM_ABBREVIATION'):
    """
    Clean the basketball data by removing invalid game occurrences and duplicates.
    
    :param df: DataFrame containing team box score team data.
    :param game_id_col: Name of the column containing game IDs.
    :param team_abbrev_col: Name of the column containing team abbreviations.
    :return: Cleaned DataFrame.
    """
    # create unique identifier by combining GAME_ID and TEAM_ABBREVIATION
    unique_id_col = 'UNIQUE_ID'
    df[unique_id_col] = df[game_id_col].astype(str) + '_' + df[team_abbrev_col]

    # drop duplicates based on the unique identifier
    df = df.drop_duplicates(unique_id_col)

    # drop the unique identifier column
    df = df.drop(unique_id_col, axis=1)
    
    # count the occurrences of each GAME_ID
    game_id_counts = df[game_id_col].value_counts()

    # filter GAME_IDs that occur exactly twice
    game_ids_to_keep = game_id_counts[game_id_counts == 2].index.tolist()

    # keep rows in DataFrame where GAME_ID occurs twice
    cleaned_df = df[df[game_id_col].isin(game_ids_to_keep)]

    return cleaned_df


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
    home_games_df = df[df['MATCHUP'].str.contains(' vs. ')].rename(
        columns={col: f'HOME_{col}' for col in df.columns if col not in non_stats_cols}
    ).drop('MATCHUP', axis=1)

    # filter for away games and rename columns
    away_games_df = df[df['MATCHUP'].str.contains(' @ ')].rename(
        columns={col: f'AWAY_{col}' for col in df.columns if col not in non_stats_cols}
    ).drop('MATCHUP', axis=1)

    # drop the non-stats columns from the away DataFrame
    away_games_df.drop(['SEASON_ID', 'GAME_DATE'], axis=1, inplace=True)

    # merge home and away DataFrames on 'GAME_ID'
    wide_df = pd.merge(home_games_df, away_games_df, on='GAME_ID')

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
    df['GAME_RESULT'] = df[home_wl_col].apply(lambda x: 1 if x == 'W' else 0)

    # calculate the total points scored
    df['TOTAL_PTS'] = df[home_pts_col] + df[away_pts_col]

    # calculate the score difference
    df['PLUS_MINUS'] = df[home_pts_col] - df[away_pts_col]

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
    prefix = 'HOME_' if 'HOME' in team_col else 'AWAY_'

    # filter the stats columns based on the prefix
    filtered_stats_cols = [col for col in stats_cols if col.startswith(prefix)]
    
    # sort data by team and time
    sorted_df = df.sort_values(by=[team_col, 'GAME_DATE']).set_index('GAME_ID')

    # calculate rolling averages for each statistic
    rolling_stats = (sorted_df.groupby(team_col)[filtered_stats_cols]
                     .rolling(window=window_size, min_periods=min_obs)
                     .mean()
                     .round(3)
                     .shift(1) # lag of 1 to exclude current value from average
                     .add_prefix('ROLL_'))

    # reset the index for merging
    rolling_stats.reset_index(inplace=True)
    
    return rolling_stats


#################################################################################
##### Function to calculate rolling average statistics and add to a DataFrame
#################################################################################
def process_rolling_stats(df, stats_cols, window_size, min_obs):
    """
    Process the DataFrame to add rolling statistics for home and away teams.

    :param df: The original DataFrame.
    :param stats_cols: List of columns for rolling statistics.
    :param window_size: The size of the rolling window.
    :param min_obs: Minimum number of observations for rolling calculation.
    :return: DataFrame with added rolling statistics.
    """
    # calculate rolling stats for home and away teams
    rolling_home_stats = calculate_rolling_stats(df, 'HOME_TEAM_NAME', stats_cols, window_size, min_obs)
    rolling_away_stats = calculate_rolling_stats(df, 'AWAY_TEAM_NAME', stats_cols, window_size, min_obs)

    # merge the rolling stats into the original DataFrame
    final_df = df.merge(rolling_home_stats.drop('HOME_TEAM_NAME', axis=1), how='left', on='GAME_ID')
    final_df = final_df.merge(rolling_away_stats.drop('AWAY_TEAM_NAME', axis=1), how='left', on='GAME_ID')

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
    df['nth_game'] = df.groupby(team_col)['GAME_DATE'].rank(method='first').astype(int)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axs = axs.ravel()

    for i, col in enumerate(plot_cols):
        if i < len(axs):
            for team in df[team_col].unique():
                team_data = df[df[team_col] == team]
                axs[i].plot(team_data['nth_game'], team_data[col], label=team, alpha=0.3)
            axs[i].set_title(col)
            axs[i].set_xlabel('n-th Game')
            axs[i].set_ylabel(col)

    for i in range(len(plot_cols), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    

#################################################################################
##### Function to scale data for modeling
#################################################################################
def scale_data(features, target, scaler='minmax', scale_target=False):
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
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
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
        target_array = target.values.reshape(-1, 1) if len(target.shape) == 1 else target
        scaled_data = np.concatenate([scaled_features, target_array], axis=1)
        # add the target column name
        col_names.append(target.name)
        
    # convert scaled data back to DataFrame, assign column names, and reset index
    scaled_data_df = pd.DataFrame(scaled_data, columns=col_names)
    scaled_data_df = scaled_data_df.reset_index(drop=True)
    
    return scaled_data_df


#################################################################################
##### Function to create a rolling window time series split
#################################################################################
def rolling_window_ts_split(df, train_size, test_size, ensure_diversity=False, target_col=None):
    """
    Generate indices to split data into training and test sets for rolling window time series cross-validation.
    
    Optionally ensures that the initial training set includes a diversity of classes for logistic regression training.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing both features and the target column.
    - train_size (int): The size of the training set for each split.
    - test_size (int): The size of the test set for each split.
    - ensure_diversity (bool): If True, checks to ensure the initial training set includes at least two classes. Default is False.
    - target_col (str): The name of the target column to check for class diversity. Required if ensure_diversity is True.
    
    Yields:
    - train_indices (np.array): Indices for the training set for each split.
    - test_indices (np.array): Indices for the test set for each split.
    
    Raises:
    - ValueError: If the dataset is not large enough for the specified train and test sizes, or if ensure_diversity is True but the initial training set does not include both classes.
    """
    import numpy as np
    
    if ensure_diversity and target_col is not None:
        # Ensure initial training set includes both classes, adjust train_size if necessary
        target_values = df[target_col].values
        unique_classes = np.unique(target_values[:train_size])
        if len(unique_classes) < 2:
            raise ValueError("Initial training set does not include both classes.")
    
    if len(df) < train_size + test_size:
        raise ValueError("The dataset is not large enough for the specified train and test sizes.")

    indices = np.arange(len(df))
    max_start_index = len(df) - train_size - test_size + 1

    for start_index in range(max_start_index):
        yield indices[start_index: start_index + train_size], indices[start_index + train_size: start_index + train_size + test_size]
        
        
#################################################################################
##### Function to create an expanding window time series split for binary targets
#################################################################################
def expanding_window_ts_split(df, initial_train_size, test_size=1, ensure_diversity=True, target_col=None):
    """
    Generate indices to split data into training and test sets for expanding window time series cross-validation,
    with an option to ensure that the initial training set includes a diversity of classes.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to be split, which should contain both features and the target column.
    - initial_train_size (int): The initial size of the training set, which will expand in each subsequent split.
    - test_size (int): The size of the test set for each split. Default is 1.
    - ensure_diversity (bool): If True, adjusts the initial_train_size to ensure the training set starts with both classes present, if the target column's diversity allows for it. Default is True.
    - target_col (str): The name of the target column to check for class diversity. Required if ensure_diversity is True.
    
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
        raise ValueError("Dataset is not large enough for the specified initial train size and test size.")

    indices = np.arange(len(df))
    for start_index in range(initial_train_size, len(df) - test_size + 1):
        train_indices = indices[:start_index]
        test_indices = indices[start_index: start_index + test_size]
        yield train_indices, test_indices
        
        