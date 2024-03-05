# SPAM Data

Spam Data library handles data engineer needs for the SPAM Capstone project, notably:

* Interacting with apis to fetch and retrieve data
  * NBA API
  * Odds Data
* Feature Engineering to generate training data
  * Rolling window
  * Opponent impact
  * etc
* Cacheing data to reduce internet traffic
  * Merging in fresh data? 

## Design

Split primarily into two sections API and feature_engineering

### API

This section pulls down information from external apis and optionally stores it within the cache.

To add or see what is available look in the `cache_filler.py` file. This has a dictionary mapping a `csv` file to a function. That function is responsible for pulling down that csv file and saving it in the cache. `load_or_fetch_from_cache` handles either calling the function or loading from cache.

Therefore, say you want to pull advanced stats for the last 2 years. You can make a new function in the `team_stats.py` file that execute the api calls and collects all the data you want and returns in. Then in cache filler add an entry to the dictionary that maps a file name, something like "advanced_stats_2_years" to the function you just made and clients will be able to get that data using `load_or_fetch_from_cache(your_filename)`

May change this in the future but it will always be similar, make a function that gets your data, and it will automagically get cached

### Feature Engineering

This section uses the `API` section, and particularly the `spam_data.api.cache_filler` methods to retrieve data from API/cache, perform data manipulations, and create datasets.

At the top level, the `build_dataset` in `spam_data.feature_engineering.__init__` is the primary way to "get a dataset/dataframe". This method works by creating the base dataset, which is the games with their targets, and then merging in `FeatureGroups` that are passed in as parameters.

`FeatureGroups` are classes with two methods, `process_data` and `merge_data_to_base`. `process_data` loads any required data from `api` (usually through `load_or_fetch_from_cache`), processes it, i.e. drops certain columns, rearranges for matchups etc.

`merge_data_to_base` joins the data for this feature group to the supplied dataframe, generally through joins on `GAME_ID` or `TEAM_IDs`

## Extending this Package

Generally the goal of any additions to this package are to make a `FeatureGroup` subclass that adds new features to a dataset. This `FeatureGroup` can then be supplied in the feature_groups argument of `spam_data.feature_engineering build_dataset` to generate a dataframe with those features included. This may also require changes to the `api` section to load data from nba_api and cache it etc.

## Setup

See setup in the ../README.md file

`pdm install` should install all the needed deps to use the library as a client

### Optional installation

* `pdm install -d` installs [black code formatter](https://github.com/psf/black)
* `pdm install -G eda` installs jupyter support
