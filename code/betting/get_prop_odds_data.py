import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import os
from collections import defaultdict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
PROP_ODDS_API_KEY = os.environ['PROP_ODDS_API_KEY']
REPO_ROOT = os.environ['REPO_ROOT']

DATA = f'{REPO_ROOT}/data/prop_odds_api'
MONEYLINE = f'{DATA}/moneyline.csv'
PLAYER_PAR = f'{DATA}/player_par.csv'
SPREAD = f'{DATA}/spread.csv'
GAMES = f'{DATA}/prop_odds_games.csv'

def prop_odds_api_req(endpoint, **kwargs):
    base_url = 'https://api.prop-odds.com'
    params = {'api_key': PROP_ODDS_API_KEY, 'tz':'America/New_York'}
    params.update(kwargs)
    x = requests.get(f'{base_url}{endpoint}', params=params)
    return x


def extract_moneyline(x):
    gid = x['game_id']
    
    res = []
    for s in x['odds'].get('sportsbooks', []):
        bookie = s['bookie_key']
        for o in s['market']['outcomes']:
            res.append(
                (gid, bookie, o['timestamp'], o['handicap'], o['odds'], o['name'])
            )
            
    pd.DataFrame(
        res, 
        columns=['game_id', 'bookie', 'timestamp', 'handicap', 'odds', 'name']
    ).to_csv(
        MONEYLINE, 
        index=None, 
        mode='a+',
        header = not os.path.exists(MONEYLINE)
    )

    
def extract_spread(x):
    gid = x['game_id']
    
    res = []
    for s in x['odds'].get('sportsbooks', []):
        bookie = s['bookie_key']
        for o in s['market']['outcomes']:
            res.append(
                (gid, bookie, o['timestamp'], o['handicap'], o['odds'], o['name'])
            )
            
    pd.DataFrame(
        res, 
        columns=['game_id', 'bookie', 'timestamp', 'handicap', 'odds', 'name']
    ).to_csv(
        SPREAD, 
        index=None, 
        mode='a+',
        header = not os.path.exists(SPREAD)
    )

    
def extract_par(x):
    gid = x['game_id']
    
    res = []
    for s in x['odds'].get('sportsbooks', []):
        bookie = s['bookie_key']
        for o in s['market']['outcomes']:
            res.append(
                (gid, bookie, o['timestamp'], o['handicap'], o['odds'], o['name'], o['participant'], o['participant_name'])
            )
            
    pd.DataFrame(
        res, 
        columns=['game_id', 'bookie', 'timestamp', 'handicap', 'odds', 'name', 'participant', 'participant_name']
    ).to_csv(
        PLAYER_PAR, 
        index=None, 
        mode='a+',
        header = not os.path.exists(PLAYER_PAR)
    )

    
def extract_odds(x):
    if x['market'] == 'moneyline':
        return extract_moneyline(x)
    if x['market'] == 'spread':
        return extract_spread(x)
    if x['market'] == 'player_assists_points_rebounds_over_under':
        return extract_par(x)
    

def get_odds_data(start_date, end_date):
    games_df = pd.read_csv(GAMES)
    seen_dates = games_df.date.unique()

    games = []
    
    while start_date < end_date:
        if start_date.strftime('%Y-%m-%d') not in seen_dates:
            res = prop_odds_api_req('/beta/games/nba', date=start_date.strftime('%Y-%m-%d'))
            games.append((res.status_code, json.loads(res.text)))
        start_date = start_date + relativedelta(days=1)

    new_game_ids = []
    for x in games:
        if x[0] != 200:
            print(x)
            continue

        all_games = []
        date = x[1]['date']
        for g in x[1]['games']:
            all_games.append((date, g['game_id'], g['away_team'], g['home_team'], g['start_timestamp']))

        temp_df = pd.DataFrame(
            all_games,
            columns = ['date', 'game_id', 'away_team', 'home_team', 'start_timestamp']
        )

        temp_df.to_csv(
            GAMES,
            mode = 'a+',
            index = None,
            header = not os.path.exists(GAMES)
        )

        new_game_ids.extend(temp_df.game_id.values)

    games_df = pd.read_csv(GAMES)
    seen_game_ids = games_df.game_id.unique()

    # get odds from prop odds api

    moneyline_df = pd.read_csv(MONEYLINE)
    par_df = pd.read_csv(PLAYER_PAR)
    spread_df = pd.read_csv(SPREAD)

    odds_game_ids = list(set(
        list(moneyline_df.game_id.unique()) + 
        list(par_df.game_id.unique()) + 
        list(spread_df.game_id.unique())
    ))

    odds = []

    for game_id in (set(seen_game_ids) - set(odds_game_ids)):
        for market in ['moneyline', 'spread', 'player_assists_points_rebounds_over_under']:
            res = prop_odds_api_req(f'/v1/odds/{game_id}/{market}')
            odds.append((game_id, market, res))

    if len(odds) > 0:
        with open(f'{DATA}/prop_odds_odds.json', 'w+') as f:
            temp = {
                'odds':[
                    {'game_id':x[0], 'market':x[1], 'odds':json.loads(x[2].text)} 
                    for x in odds
                ]
            }
            f.write(json.dumps(temp))

        for x in temp['odds']:
            extract_odds(x)

start_date = datetime.strptime('2023-08-01', '%Y-%m-%d')
end_date = datetime.strptime('2024-03-06', '%Y-%m-%d')
get_odds_data(start_date, end_date)
                  
      
                