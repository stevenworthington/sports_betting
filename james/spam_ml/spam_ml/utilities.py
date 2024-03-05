
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
    df = df[df['MIN'] >= 238]

    # convert GAME_DATE to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    # initialize an empty DataFrame for the cleaned data
    df_reg_seasons = pd.DataFrame()
    
    # map each game to its season and replace SEASON_ID
    for i, (start_date, end_date) in enumerate(zip(season_start_dates, season_end_dates)):
        # select games within the current season date range
        mask = (df['GAME_DATE'] >= start_date) & (df['GAME_DATE'] <= end_date)
        df_season = df.loc[mask].copy()
        if not df_season.empty:
            df_season['SEASON_ID'] = season_labels[i]
            df_reg_seasons = pd.concat([df_reg_seasons, df_season])
    
    # print the number of games in each season using the SEASON_ID labels
    for label in sorted(set(season_labels)):
        num_games = df_reg_seasons[df_reg_seasons['SEASON_ID'] == label]['GAME_ID'].nunique()
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
    
    season_game_counts = wide_df.groupby('SEASON_ID')['GAME_ID'].nunique() 
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
    df['GAME_RESULT'] = df[home_wl_col].apply(lambda x: 1 if x == 'W' else 0)

    # calculate the total points scored
    df['TOTAL_PTS'] = df[home_pts_col] + df[away_pts_col]
    
    # rename the home score difference
    df.rename(columns={'HOME_PLUS_MINUS': 'PLUS_MINUS'}, inplace=True)
    
    # drop the away score difference
    df.drop('AWAY_PLUS_MINUS', axis=1, inplace=True)
    
    return df


#################################################################################
##### Function to calculate rolling average statistics
#################################################################################
def calculate_rolling_stats(df, team_col, stats_cols, window_size, min_obs,
                            stratify_by_season=True, exclude_initial_games=0):
    """
    Calculate rolling statistics for a given team, optionally stratifying by season, and excluding the initial games from the rolling average calculations.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the team data.
    - team_col (str): Column name identifying the team within the DataFrame.
    - stats_cols (list): List of statistic columns to calculate rolling averages for.
    - window_size (int): Number of games over which to calculate the rolling average.
    - min_obs (int): Minimum number of observations required to calculate the rolling average.
    - stratify_by_season (bool): Whether to reset rolling calculations at the start of each season.
    - exclude_initial_games (int): Number of initial games in each season to exclude from the rolling calculations.

    Returns:
    - pd.DataFrame: DataFrame with rolling statistics, excluding initial games as specified.
    """
    import pandas as pd
    import numpy as np
    
    # determine whether to use 'HOME' or 'AWAY' stats
    prefix = 'HOME_' if 'HOME' in team_col else 'AWAY_'

    # filter the stats columns based on the prefix
    filtered_stats_cols = [col for col in stats_cols if col.startswith(prefix)]
    
    # ensure data is sorted by team, season (if stratified), and date for accurate rolling calculation
    # set GAME_ID and GAME_DATE as the indices to preserve them through the rolling operation
    sort_cols = [team_col, 'SEASON_ID', 'GAME_DATE'] if stratify_by_season else [team_col, 'GAME_DATE']
    df_sorted = df.sort_values(by=sort_cols).set_index(['GAME_ID', 'GAME_DATE'])

    # apply grouping for rolling calculation
    group_cols = [team_col, 'SEASON_ID'] if stratify_by_season else [team_col]
    
    # perform the rolling operation, preserving GAME_ID and GAME DATE in the index
    rolling_stats = (df_sorted.groupby(group_cols)[filtered_stats_cols]
                              .rolling(window=window_size, min_periods=min_obs)
                              .mean()
                              .round(3)
                              .groupby(group_cols) # need to groupby again for shift
                              .shift(1)  # lag to exclude the current game from the rolling average
                              .add_prefix('ROLL_')
                              .reset_index() # reset the index to convert GAME_ID and GAME DATE back into columns
                     )

    # adjust 'filtered_stats_cols' to include the 'ROLL_' prefix
    rolled_filtered_stats_cols = ['ROLL_' + col for col in filtered_stats_cols]
    
    # exclude initial games if specified by setting the first n rows
    # to NaN within each group for the statistical columns
    if exclude_initial_games > 0:
        
        def set_initial_games_to_nan(group):
            group.loc[group.index[:exclude_initial_games], rolled_filtered_stats_cols] = np.nan
            return group

        # apply the function to each group
        rolling_stats = rolling_stats.groupby([team_col] + (['SEASON_ID'] if stratify_by_season else [])).apply(set_initial_games_to_nan)

    return rolling_stats


#################################################################################
##### Function to calculate rolling difference average statistics
#################################################################################
def calculate_rolling_diff_stats(df, team_col, stats_cols, window_size, min_obs,
                                 stratify_by_season=True, exclude_initial_games=0):
    """
    Calculate rolling average of the over/under performance in a given statistic for a given team compared to their opponent, 
    optionally stratifying by season, and excluding the initial games from the rolling average calculations.
    Example: Average the last n games of the statistics for (HOME_PTS - AWAY_PTS) or (HOME_FG_PCT - AWAY_FG_PCT).
    This will give us intuition on how the team is performing against its opponents. 

    Parameters:
    - df (pd.DataFrame): DataFrame containing the team data.
    - team_col (str): Column name identifying the team within the DataFrame.
    - stats_cols (list): List of statistic columns to calculate rolling averages for.
    - window_size (int): Number of games over which to calculate the rolling average.
    - min_obs (int): Minimum number of observations required to calculate the rolling average.
    - stratify_by_season (bool): Whether to reset rolling calculations at the start of each season.
    - exclude_initial_games (int): Number of initial games in each season to exclude from the rolling calculations.

    Returns:
    - pd.DataFrame: DataFrame with rolling statistics, excluding initial games as specified.
    """
    import pandas as pd
    import numpy as np
    
    # determine whether to use 'HOME' or 'AWAY' stats
    prefix = 'HOME_' if 'HOME' in team_col else 'AWAY_'

    # filter the stats columns based on prefix
    # Stats columns for Home, Away, and both will be required for calculation
    filtered_stats_cols_home = [col for col in stats_cols if col.startswith('HOME_')]
    filtered_stats_cols_away = [col for col in stats_cols if col.startswith('AWAY_')]
    filtered_stats_cols_both = [col for col in stats_cols if col.startswith('HOME_') or col.startswith('AWAY_')]

    # ensure data is sorted by team, season (if stratified), and date for accurate rolling calculation
    # set GAME_ID and GAME_DATE as the indices to preserve them through the rolling operation
    sort_cols = [team_col, 'SEASON_ID', 'GAME_DATE'] if stratify_by_season else [team_col, 'GAME_DATE']
    df_sorted = df.sort_values(by=sort_cols).set_index(['GAME_ID', 'GAME_DATE'])

    # apply grouping for rolling calculation
    group_cols = [team_col, 'SEASON_ID'] if stratify_by_season else [team_col]
    
    # perform the rolling operation, preserving GAME_ID and GAME DATE in the index
    if prefix == 'HOME_':

        rolling_stats = (df_sorted.groupby(group_cols)[filtered_stats_cols_both]
                                  .diff(axis=1, periods=-18)[filtered_stats_cols_home]
                                  .rolling(window=window_size, min_periods=min_obs)
                                  .mean()
                                  .round(3)
                                  .groupby(group_cols) # need to groupby again for shift
                                  .shift(1)  # lag to exclude the current game from the rolling average
                                  .add_prefix('ROLLDIFF_')
                                  .reset_index() # reset the index to convert GAME_ID and GAME_DATE back into columns
                        )

    if prefix == 'AWAY_':

        rolling_stats = (df_sorted.groupby(group_cols)[filtered_stats_cols_both]
                                  .diff(axis=1, periods=18)[filtered_stats_cols_away]
                                  .rolling(window=window_size, min_periods=min_obs)
                                  .mean()
                                  .round(3)
                                  .groupby(group_cols) # need to groupby again for shift
                                  .shift(1)  # lag to exclude the current game from the rolling average
                                  .add_prefix('ROLLDIFF_')
                                  .reset_index() # reset the index to convert GAME_ID and GAME_DATE back into columns
                        )

    # adjust 'filtered_stats_cols' to include the 'ROLLDIFF_' prefix
    if prefix == 'HOME_':
        rolled_filtered_stats_cols = ['ROLLDIFF_' + col for col in filtered_stats_cols_home]

    if prefix == 'AWAY_':
        rolled_filtered_stats_cols = ['ROLLDIFF_' + col for col in filtered_stats_cols_away]

    # exclude initial games if specified by setting the first n rows
    # to NaN within each group for the statistical columns
    if exclude_initial_games > 0:
        
        def set_initial_games_to_nan(group):
            group.loc[group.index[:exclude_initial_games], rolled_filtered_stats_cols] = np.nan
            return group

        # apply the function to each group
        rolling_stats = rolling_stats.groupby([team_col] + (['SEASON_ID'] if stratify_by_season else [])).apply(set_initial_games_to_nan)

    return rolling_stats

    
#################################################################################
##### Function to calculate rolling average statistics and add to a DataFrame
#################################################################################
def process_rolling_stats(df, stats_cols, target_cols, window_size, min_obs,  
                          stratify_by_season=True, exclude_initial_games=0):
    """
    Process the DataFrame to add rolling statistics for home and away teams, with an option to exclude the initial games from the rolling calculation. Optionally allows for rolling calculations to reset at the start of each new season or be contiguous across seasons. Merges rolling statistics into a subset of the original DataFrame that includes specified target columns, using GAME_ID as the unique identifier.

    Parameters:
    - df: The original DataFrame.
    - stats_cols: List of columns for rolling statistics.
    - target_cols: List of target column names to be included in the final DataFrame.
    - window_size: The size of the rolling window.
    - min_obs: Minimum number of observations for rolling calculation.
    - stratify_by_season: If True, calculate rolling stats separately for each season. If False, calculate rolling stats across seasons.
    - exclude_initial_games: Number of initial games to exclude from the rolling calculations.
    
    Returns:
    - DataFrame with added rolling statistics merged into the specified target columns.
    """
    # create a subset of the original DataFrame that includes only the target columns
    df_subset = df[['GAME_ID'] + target_cols].drop_duplicates()

    # calculate rolling stats for home and away teams with optional season stratification and exclusion of initial games
    rolling_home_stats = calculate_rolling_stats(df, 'HOME_TEAM_NAME', stats_cols, window_size, min_obs, stratify_by_season, exclude_initial_games)
    rolling_away_stats = calculate_rolling_stats(df, 'AWAY_TEAM_NAME', stats_cols, window_size, min_obs, stratify_by_season, exclude_initial_games)

    # merge the rolling stats into the subset DataFrame using GAME_ID as the merge key
    final_df = df_subset.merge(rolling_home_stats, how='left', on='GAME_ID')
    final_df = final_df.merge(rolling_away_stats.drop(['SEASON_ID', 'GAME_DATE'], axis=1), how='left', on='GAME_ID')
    
    # sort the final DataFrame by GAME_DATE from oldest to newest
    final_df = final_df.sort_values(by='GAME_DATE')
    
    return final_df


#################################################################################
##### Function to calculate rolling difference average statistics and add to a DataFrame
#################################################################################
def process_rolling_diff_stats(df, stats_cols, target_cols, window_size, min_obs,  
                               stratify_by_season=True, exclude_initial_games=0):
    """
    Process the DataFrame to add rolling statistics for home and away teams, with an option to exclude the initial games from the rolling calculation. Optionally allows for rolling calculations to reset at the start of each new season or be contiguous across seasons. Merges rolling difference statistics into a subset of the original DataFrame that includes specified target columns, using GAME_ID as the unique identifier.

    Parameters:
    - df: The original DataFrame.
    - stats_cols: List of columns for rolling statistics.
    - target_cols: List of target column names to be included in the final DataFrame.
    - window_size: The size of the rolling window.
    - min_obs: Minimum number of observations for rolling calculation.
    - stratify_by_season: If True, calculate rolling stats separately for each season. If False, calculate rolling stats across seasons.
    - exclude_initial_games: Number of initial games to exclude from the rolling calculations.
    
    Returns:
    - DataFrame with added rolling statistics merged into the specified target columns.
    """
    # create a subset of the original DataFrame that includes only the target columns
    df_subset = df[['GAME_ID'] + target_cols].drop_duplicates()

    # calculate rolling stats for home and away teams with optional season stratification and exclusion of initial games
    rolling_home_stats = calculate_rolling_diff_stats(df, 'HOME_TEAM_NAME', stats_cols, window_size, min_obs, stratify_by_season, exclude_initial_games)
    rolling_away_stats = calculate_rolling_diff_stats(df, 'AWAY_TEAM_NAME', stats_cols, window_size, min_obs, stratify_by_season, exclude_initial_games)

    # merge the rolling stats into the subset DataFrame using GAME_ID as the merge key
    final_df = df_subset.merge(rolling_home_stats, how='left', on='GAME_ID')
    final_df = final_df.merge(rolling_away_stats.drop(['SEASON_ID', 'GAME_DATE'], axis=1), how='left', on='GAME_ID')
    
    # sort the final DataFrame by GAME_DATE from oldest to newest
    final_df = final_df.sort_values(by='GAME_DATE')
    
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
##### Function to load, filter (by time), and scale data for modeling
#################################################################################
def load_and_scale_data(file_path, seasons_to_keep, training_season, feature_prefix, 
                        scaler_type='minmax', scale_target=False, csv_out=False, output_path=None):
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
    df = pd.read_csv(file_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # filter the DataFrame for the specified seasons
    df_filtered = df[df['SEASON_ID'].isin(seasons_to_keep)]
    
    # initialize the scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler must be either 'minmax' or 'standard'")
    
    # define feature names
    feature_names = [col for col in df_filtered.columns if col.startswith(feature_prefix)]
    
    # identify rows with missing values in any of the feature_names and extract non-statistical data
    missing_features_rows = df_filtered[df_filtered[feature_names].isnull().any(axis=1)]
    dropped_obs_data = missing_features_rows[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME']]
    
    # identify rows with no missing values in any of the feature_names and extract non-statistical data
    no_missing_features_rows = df_filtered[~df_filtered[feature_names].isnull().any(axis=1)]
    kept_obs_data = no_missing_features_rows[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME']]
    
    # extract non-statistical data for non-missing rows in the test set (season 2023-24)
    kept_test_set_obs = kept_obs_data[kept_obs_data['SEASON_ID'] == '2023-24']
    
    if csv_out and (output_path is not None):
        dropped_obs_data.to_csv(f"{output_path}nba_rolling_box_scores_dropped_observations.csv", index=False)
        kept_obs_data.to_csv(f"{output_path}nba_rolling_box_scores_kept_observations.csv", index=False)
        kept_test_set_obs.to_csv(f"{output_path}nba_rolling_box_scores_test_set_observations.csv", index=False)
        
    # drop rows with missing values from df_filtered
    df_filtered = df_filtered.dropna(subset=feature_names)
    
    # set GAME_DATE as the index
    df_filtered.set_index('GAME_DATE', inplace=True)
    
    # fit the scaler on features from the training season only
    training_features = df_filtered[df_filtered['SEASON_ID'] == training_season][feature_names]
    scaler.fit(training_features)

    # prepare to store scaled data for each target
    pts_scaled_df, pm_scaled_df, res_scaled_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # apply scaling to features (and optionally targets) for all specified target columns
    for target_col in ['TOTAL_PTS', 'PLUS_MINUS', 'GAME_RESULT']:
        
        # optionally scale target
        if scale_target:
            df_filtered[target_col] = scaler.transform(df_filtered[[target_col]])
        
        # select and assign scaled data for each target_col
        if target_col == 'TOTAL_PTS':
            pts_scaled_df = df_filtered[feature_names + [target_col]].copy()
        elif target_col == 'PLUS_MINUS':
            pm_scaled_df = df_filtered[feature_names + [target_col]].copy()
        elif target_col == 'GAME_RESULT':
            res_scaled_df = df_filtered[feature_names + [target_col]].copy()
    
    # print the number of unique games for each season in the filtered data
    for season_id in seasons_to_keep:
        num_games = df_filtered[df_filtered['SEASON_ID'] == season_id]['GAME_ID'].nunique()
        print(f"Season {season_id}: {num_games} games")
        
    # print total number of unique games in the filtered data
    print(f"Total number of games across sampled seasons: {df_filtered['GAME_ID'].nunique()} games")
    
    return pts_scaled_df, pm_scaled_df, res_scaled_df, kept_test_set_obs

        
#################################################################################
##### Function to create an expanding window time series split
#################################################################################
def expanding_window_ts_split(df, initial_train_size, test_size, gap_size=0, 
                              expansion_limit=None, ensure_diversity=False, target_col=None):
    """
    Generate indices for training and test sets with a specified gap and test set size, 
    considering an expansion limit for the number of predictions, and ensuring diversity in the initial training set if required.

    This function also allows for a smaller final test set to ensure that all data can be used for testing, even if the remaining data does not fit the exact test_size specification.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be split.
    - initial_train_size (int): The initial size of the training dataset.
    - test_size (int): The size of the test dataset for each expansion.
    - gap_size (int): The gap size between training and test datasets. Default is 0.
    - expansion_limit (int, optional): The limit on the number of predictions to generate.
    - ensure_diversity (bool, optional): If True, ensures the initial training set includes at least two classes.
    - target_col (str, optional): The name of the target column for checking class diversity.

    Yields:
    - train_indices, test_indices (np.array): Indices for training and test sets for each split.
    """
    import numpy as np

    if ensure_diversity and target_col is not None:
        target_values = df[target_col].values
        second_class_indices = np.where(target_values != target_values[0])[0]
        if len(second_class_indices) > 0:
            second_class_start = second_class_indices[0]
            initial_train_size = max(initial_train_size, second_class_start + 1)
        else:
            raise ValueError("The dataset does not contain enough diversity for the specified target column.")

    total_length = len(df)
    current_test_end = initial_train_size
    total_predictions = 0

    while True:
        train_end = current_test_end
        test_start = train_end + gap_size
        test_end = test_start + test_size

        # adjust test_end for the final set if necessary
        if test_end > total_length or (expansion_limit is not None and total_predictions + (test_end - test_start) > expansion_limit):
            test_end = min(total_length, test_start + (expansion_limit - total_predictions) if expansion_limit is not None else total_length)

        if test_start >= total_length:
            break

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices

        total_predictions += len(test_indices)
        current_test_end = train_end + test_size

        # break condition if we reach the expansion limit or the end of the data
        if (expansion_limit is not None and total_predictions >= expansion_limit) or test_end >= total_length:
            break
        

#################################################################################
##### Function to train models on an expanding window time series split
#################################################################################
def train_with_expanding_window(df, initial_train_size, test_size, target_col, model, gap_size=0,
                                expansion_limit=None, fitted_model=False, ensure_diversity=False):
    """
    Trains a given model using an expanding window approach on a specified DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and target variable.
    - initial_train_size (int): The initial size of the training dataset.
    - test_size (int): The size of the test dataset for each split, typically 1 for LOO CV.
    - target_col (str): The name of the target column in `df`.
    - model (model object): The instantiated model to be trained, e.g., LinearRegression() or LogisticRegression().
    - gap_size (int): The gap size between training and test datasets. Default is 0.
    - expansion_limit (int, optional): The maximum number of times the training set is expanded by 1 observation during the expanding window process. This parameter controls the total number of train-test splits generated, indirectly determining the final size of the training set. If set, the training process will stop once this limit is reached, potentially leaving some data unused. If None, the training set will expand until all but the last observation are used for training.
    - fitted_model (bool): whether to return the fitted model instance.
    - ensure_diversity (bool, optional): For logistic regression, ensures the initial training data includes both classes. Default is False.

    Returns:
    - model_outputs (list): A list of model predictions or probabilities for the test sets across all splits.
    - y_true (list): A list of the actual target values corresponding to each prediction in `model_outputs`.

    This function iterates over the dataset using an expanding window to create training and test splits, 
    trains the specified `model` on each training split, and stores the model's predictions or probabilities.
    """
    import time
    
    start_time = time.time()

    # initialize storage for model outputs and true labels
    model_outputs = []  # store predictions or probabilities
    y_true = []

    for train_indices, test_indices in expanding_window_ts_split(
        df, initial_train_size, test_size=test_size, gap_size=gap_size,
        expansion_limit=expansion_limit, ensure_diversity=ensure_diversity, 
        target_col=target_col if ensure_diversity else None):
        
        # get training and testing data for this window
        X_train = df.iloc[train_indices].drop(columns=target_col)
        y_train = df.iloc[train_indices][target_col]
        X_test = df.iloc[test_indices].drop(columns=target_col)
        y_test = df.iloc[test_indices][target_col]
        
        # train the model
        model.fit(X_train, y_train)
        
        # check if the model has the predict_proba method (i.e., likely a classifier)
        if hasattr(model, 'predict_proba'):
            # store predicted probabilities of the positive class
            proba = model.predict_proba(X_test)[:, 1]
            model_outputs.extend(list(proba.flatten()))  # ensure it's flattened
        elif hasattr(model, 'predict'):
            # predict for models that support predict (regressors and classifiers without predict_proba)
            predictions = model.predict(X_test)
            model_outputs.extend(list(predictions.flatten()))  # ensure it's flattened
        else:
            raise ValueError("Model does not support required prediction methods.")

        # store true labels for evaluation
        y_true.extend(list(y_test))

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    if fitted_model:
        return model, model_outputs, y_true
    else:
        return model_outputs, y_true


#################################################################################
##### Function to train models on a hyperparamter grid with expanding window
#################################################################################
def train_models_over_grid(df, target_col, initial_train_size, test_size, gap_size,   
                           expansion_limit, model_class, constant_params, explore_params):
    """
    Trains models over a grid of hyperparameters.

    Parameters:
    - df (pd.DataFrame): The dataset to use for training.
    - target_col (str): The name of the target column.
    - initial_train_size (int): Starting size of the training set.
    - test_size (int): Size of the test dataset for each split (LOO cross-validation).
    - gap_size (int): The gap size between training and test datasets. Default is 0.
    - expansion_limit (int): Maximum number of new training observations in expansion.
    - model_class: The class of the model to instantiate.
    - constant_params (dict): Constant parameters for the model.
    - explore_params (dict): Parameters to explore with grid search.

    Returns:
    - dict: A dictionary containing the results for each run.
    """
    import itertools
    import time

    results = {}
    keys, values = zip(*explore_params.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    start_time = time.time()

    for i, explore_param in enumerate(param_combinations):
        print('Parameters currently explored:', explore_param)
        
        # instantiate the model with combined parameters
        model = model_class(**constant_params, **explore_param)

        # train over expanding window
        model_outputs, y_true = train_with_expanding_window(
            df=df,
            initial_train_size=initial_train_size,
            test_size=test_size,
            gap_size=gap_size,
            expansion_limit=expansion_limit,
            target_col=target_col,
            model=model
        )
        
        # store outputs and true values in the results dictionary
        results[f"run_{i}"] = {
            "params": {**constant_params, **explore_param},
            "model_outputs": model_outputs,
            "y_true": y_true
        }

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return results


#################################################################################
##### Function to create a rolling window time series split
#################################################################################
def rolling_window_ts_split(df, train_size, test_size, ensure_diversity=False, 
                            target_col=None, gap_size=0, advancement_limit=None):
    """
    Generate indices to split data into training and test sets for rolling window time series cross-validation,
    with an option to include a gap between the training and testing sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing both features and the target column.
    - train_size (int): The size of the training set for each split.
    - test_size (int): The size of the test set for each split.
    - ensure_diversity (bool): If True, checks to ensure the initial training set includes at least two classes. Default is False.
    - target_col (str): The name of the target column to check for class diversity. Required if ensure_diversity is True.
    - gap_size (int, optional): Specifies the size of the gap between the training and testing sets. Default is 0.
    - advancement_limit (int, optional): Specifies the maximum number of times the training window is allowed to advance.

    Yields:
    - train_indices (np.array): Indices for the training set for each split.
    - test_indices (np.array): Indices for the test set for each split, taking into account the gap size.
    """
    import numpy as np

    if ensure_diversity and target_col is not None:
        # ensure initial training set includes both classes, adjust train_size if necessary
        target_values = df[target_col].values
        unique_classes = np.unique(target_values[:train_size])
        if len(unique_classes) < 2:
            raise ValueError("Initial training set does not include both classes.")
    
    total_length = len(df)
    if total_length < train_size + test_size + gap_size:
        raise ValueError("The dataset is not large enough for the specified configuration.")

    indices = np.arange(total_length)
    max_start_index = total_length - train_size - test_size - gap_size + 1
    actual_advancements = min(advancement_limit, max_start_index) if advancement_limit is not None else max_start_index

    for start_index in range(0, actual_advancements):
        train_end_index = start_index + train_size
        test_start_index = train_end_index + gap_size
        if test_start_index + test_size > total_length:
            break  # ensure we don't exceed the dataset size
        yield indices[start_index: train_end_index], indices[test_start_index: test_start_index + test_size]


#################################################################################
##### Function to train models on an rolling window time series split
#################################################################################
def train_with_rolling_window(df, train_size, test_size, target_col, model, ensure_diversity=False, 
                              gap_size=0, advancement_limit=None, fitted_model=False):
    """
    Trains a specified model using a rolling window approach on a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing both features and the target variable.
    - train_size (int): The size of the training set for each split.
    - test_size (int): The size of the test set for each split, typically 1 for leave-one-out cross-validation.
    - target_col (str): The name of the target column in `df`.
    - model (model object): The instantiated model to be trained. This can be any model that conforms to the scikit-learn model interface, such as instances of `LinearRegression` or `LogisticRegression`.
    - ensure_diversity (bool, optional): Indicates whether to ensure the initial training data includes a diverse set of classes for classification tasks. This is primarily relevant for logistic regression and similar models where class diversity in the training set might impact model training. Default is False.
    - gap_size (int, optional): Specifies the size of the gap between the training and testing sets. Default is 0.
    - advancement_limit (int, optional): Specifies the maximum number of times the training window is allowed to advance during the rolling window process. This parameter effectively limits the number of train-test splits generated, allowing for control over the number of models trained and tested, which can be useful for large datasets or for limiting computational expense. If not set, or if set to None, the window will advance until it reaches the end of the dataset, making use of all possible train-test splits given the `train_size` and `test_size` parameters. Setting this parameter helps in focusing the training and testing process on a specific subset of the dataset, potentially leaving some data unused at the end of the dataset.
    - fitted_model (bool): whether to return the fitted model instance.

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
        df, train_size, test_size, ensure_diversity=ensure_diversity, 
        target_col=target_col if ensure_diversity else None, gap_size=gap_size,
        advancement_limit=advancement_limit):
        
        # get training and testing data for this window
        X_train = df.iloc[train_indices].drop(columns=target_col)
        y_train = df.iloc[train_indices][target_col]
        X_test = df.iloc[test_indices].drop(columns=target_col)
        y_test = df.iloc[test_indices][target_col]
        
        # train the model
        model.fit(X_train, y_train)
        
        # check if the model has the predict_proba method (i.e., likely a classifier)
        if hasattr(model, 'predict_proba'):
            # store predicted probabilities of the positive class
            proba = model.predict_proba(X_test)[:, 1]
            model_outputs.extend(proba)
        elif hasattr(model, 'predict'):
            # for models that support predict (regressors and classifiers without predict_proba)
            predictions = model.predict(X_test)
            model_outputs.extend(predictions)
        else:
            raise ValueError("Model does not support required prediction methods.")

        # store true labels for evaluation
        y_true.extend(y_test)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    if fitted_model:
        return model, model_outputs, y_true
    else:
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
    from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score
    import numpy as np

    metrics = {}  # dictionary to store calculated metrics

    unique_values = np.unique(y_true)
    if len(unique_values) == 2:  # binary classification
        pred_labels = [1 if p > threshold else 0 for p in model_outputs]
        metrics['pred_labels'] = pred_labels
        metrics['average_accuracy'] = accuracy_score(y_true, pred_labels)
        metrics['overall_auc'] = roc_auc_score(y_true, model_outputs)
        metrics['average_f1_score'] = f1_score(y_true, pred_labels)
        if verbose:
            print(f"Classification Metrics:\n- Average Accuracy: {metrics['average_accuracy']:.2f}\n- Overall AUC: {metrics['overall_auc']:.2f}\n- Average F1 Score: {metrics['average_f1_score']:.2f}")
    else:  # regression
        metrics['average_rmse'] = mean_squared_error(y_true, model_outputs, squared=False)
        y_true_mean = np.full_like(y_true, np.mean(y_true))
        metrics['null_rmse'] = mean_squared_error(y_true, y_true_mean, squared=False)
        if verbose:
            print(f"Regression Metrics:\n- Average RMSE: {metrics['average_rmse']:.2f}\n- Null RMSE: {metrics['null_rmse']:.2f}")

    return metrics


#################################################################################
##### Function to calculate performance metrics from trained models
#################################################################################
def compile_results_to_dataframe(results, calculate_metrics_function=calculate_metrics, verbose=False):
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
                                             each value is the metric's value.
    
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
        params = run_data['params']
        y_true = run_data['y_true']
        model_outputs = run_data['model_outputs']
        
        # calculate metrics for this run
        metrics = calculate_metrics_function(y_true, model_outputs, verbose=verbose)
        
        # combine params and metrics into a single dictionary
        combined_data = {**params, **metrics}
        
        # add unique identifier for the run
        combined_data['run_id'] = run_id
        
        # append the combined data to our list
        data_for_df.append(combined_data)

    # create a DataFrame from the list of combined data
    metrics_df = pd.DataFrame(data_for_df)

    # reorder the DataFrame columns
    columns_order = ['run_id'] + sorted([col for col in metrics_df.columns if col != 'run_id'])
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
    params_to_exclude = ['run_id', 'average_rmse', 'null_rmse', 'average_accuracy', 
                         'average_f1_score', 'overall_auc', 'pred_labels']

    # determine whether to find the min or max value based on the metric
    if metric == 'average_rmse':
        best_row_idx = df[metric].idxmin()
    elif metric in ['average_accuracy', 'overall_auc', 'average_f1_score']:
        best_row_idx = df[metric].idxmax()

    # find the row with the optimal performance metric
    best_row = df.loc[best_row_idx]

    # create the dictionary of best parameters, excluding specified columns
    best_params = {param: best_row[param] for param in best_row.index if param not in params_to_exclude}

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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))
    

#################################################################################
##### Function to plot model performance for continuous outcomes with expanding window
#################################################################################
def plot_model_performance(title, y_label, metric, df, model_outputs, target_col, 
                           initial_train_size, skip_start=0, expansion_limit=None, test_size=1):
    """
    Plots the observed vs. predicted values and includes a performance metric on the plot, with adjustments for skip_start.

    Parameters:
    - title (str): The title for the plot.
    - y_label (str): The label for the y-axis.
    - metric (dict element): The dictionary element for the performance metric to display.
    - df (pd.DataFrame): The DataFrame containing the observed data.
    - model_outputs (list or np.array): The model's predictions.
    - target_col (str): The name of the target column in `df`.
    - initial_train_size (int): The initial size of the training dataset.
    - skip_start (int): The number of initial testing observations to skip.
    - expansion_limit (int or None): The maximum number of times the training set is expanded.
    - test_size (int): The size of the test dataset for each split.
    """
    import matplotlib.pyplot as plt
    
    # adjust the starting index of observed values based on skip_start
    start_index_adjusted = initial_train_size + skip_start * test_size

    # set end_index based on whether an expansion_limit is specified
    if expansion_limit is not None:
        end_index = start_index_adjusted + expansion_limit * test_size
    else:
        # use the length of model_outputs to determine end_index when no limit is specified
        end_index = start_index_adjusted + len(model_outputs) * test_size

    # adjust the observed_values slice accordingly
    observed_values = df[target_col][start_index_adjusted:end_index]
    
    # ensure lengths are the same (for sanity check)
    assert len(model_outputs) == len(observed_values), "Length mismatch between predictions and observed values."

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(observed_values.index, observed_values, label='Observed', color='blue', alpha=0.5)
    plt.plot(observed_values.index, model_outputs, label='Predicted', color='red', alpha=0.5)
    plt.title(title)
    plt.xlabel(r'$n^{th}$ Game')
    plt.ylabel(y_label)
    plt.text(x=0.72, y=0.92, s=f"Average RMSE: {metric:.2f}", 
             transform=plt.gca().transAxes, ha='left', va='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.show()
    

#################################################################################
##### Function to plot model performance for continuous outcomes with rolling window
#################################################################################
def plot_model_performance_rolling(title, y_label, metric, df, model_outputs, target_col,
                                   train_size, skip_start=0, advancement_limit=None, test_size=1):
    """
    Plots observed vs. predicted values for a model using a rolling window approach and includes a performance metric.
    Adjusts for skipped observations at the start of the rolling window process.

    Parameters:
    - title (str): Title for the plot.
    - y_label (str): Label for the y-axis.
    - metric (dict element): The dictionary element for the performance metric to display.
    - df (pd.DataFrame): DataFrame containing the observed data.
    - model_outputs (list or np.array): Model's predictions.
    - target_col (str): Name of the target column in `df`.
    - train_size (int): Initial size of the training dataset.
    - skip_start (int): Number of initial observations to skip in the rolling window process.
    - advancement_limit (int or None): Maximum number of times the training set is expanded.
    - test_size (int): Size of the test dataset for each split.
    """
    import matplotlib.pyplot as plt

    # Adjust start index for plotting to account for skip_start
    plot_start_index = train_size + skip_start * test_size

    if advancement_limit is not None:
        end_index = plot_start_index + advancement_limit * test_size
    else:
        end_index = plot_start_index + len(model_outputs) * test_size

    observed_values = df[target_col][plot_start_index:end_index]

    assert len(model_outputs) == len(observed_values), "Length mismatch between predictions and observed values."

    plt.figure(figsize=(12, 4))
    plt.plot(observed_values.index, observed_values, label='Observed', color='blue', alpha=0.5)
    plt.plot(observed_values.index, model_outputs, label='Predicted', color='red', alpha=0.5)
    plt.title(title)
    plt.xlabel(r'$n^{th}$ Game')
    plt.ylabel(y_label)
    plt.text(x=0.72, y=0.92, s=f"Average RMSE: {metric:.2f}", 
             transform=plt.gca().transAxes, ha='left', va='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.show()
    

#################################################################################
##### Function to plot classification performance
#################################################################################
def plot_classification_performance(y_true, model_outputs, pred_labels, cm_title='Confusion Matrix',  
                                    roc_title='ROC Curve', overall_title='Classification Performance'):
    """
    Plots the confusion matrix and ROC curve for classification model results side by side, with an overall title above the plots.

    Parameters:
    - y_true (array-like): True labels.
    - model_outputs (array-like): Model probability outputs for the positive class.
    - pred_labels (array-like): Predicted labels.
    - cm_title (str): Title for the confusion matrix plot.
    - roc_title (str): Title for the ROC curve plot.
    - overall_title (str): Overall title for both plots.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    # plot Confusion Matrix
    cm = confusion_matrix(y_true, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues', ax=axes[0])
    axes[0].set_title(cm_title)
    
    # plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, model_outputs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(roc_title)
    axes[1].legend(loc="lower right")
    
    plt.suptitle(overall_title, fontsize=14)  # set overall title for both plots
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # adjust layout to make room for the overall_title
    plt.show()


