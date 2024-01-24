
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
    
 