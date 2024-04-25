import pandas as pd
from functools import partial
from statistics import NormalDist
from datetime import datetime
from simulate_wagers import get_other_team
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
REPO_ROOT = os.environ['REPO_ROOT']
PREDICTIONS_DIR = '../../data/model_predictions/'
ODDS_DIR = '../../data/prop_odds_api/'
GAME_DATA = f'{REPO_ROOT}/data/original/'

ODDS = {
    'moneyline': pd.read_csv(ODDS_DIR + 'moneyline.csv'),
    'spread': pd.read_csv(ODDS_DIR + 'spread.csv')
}
ODDS_GAMES = pd.read_csv(ODDS_DIR + 'prop_odds_games.csv')
BOX_SCORES = pd.read_csv(GAME_DATA + 'nba_games_box_scores_2022_2024.csv', usecols=['TEAM_NAME', 'GAME_DATE', 'PTS'])


def get_game_id(home_team, away_team, game_date):
    game_id = ODDS_GAMES[
        (ODDS_GAMES.away_team == away_team)
        & (ODDS_GAMES.home_team == home_team)
        & (ODDS_GAMES.date == game_date)
    ].game_id.iloc[0]
    return game_id


def get_opening_odds(available_odds, team):
    odds = available_odds[available_odds.name == team]
    if len(odds) > 0:
        odds = (
            odds.iloc[0].handicap,
            odds.iloc[0].odds
        )
    else:
        odds = (None, None)
    return odds

def opening_line(row, betting_market='spread', bookie='betmgm'):    
    game_id = get_game_id(row.HOME_TEAM_NAME, row.AWAY_TEAM_NAME, row.GAME_DATE)
    available_odds = ODDS[betting_market][
        (ODDS[betting_market].game_id==game_id) 
        & (ODDS[betting_market].bookie==bookie)
    ]
    home_odds = get_opening_odds(available_odds, row.HOME_TEAM_NAME)
    away_odds = get_opening_odds(available_odds, row.AWAY_TEAM_NAME)
    return home_odds[0], home_odds[1], away_odds[0], away_odds[1]
    
def payout_odds(american_odds):
    # win % required to break even
    # 300 odds -> payout 3:1,  -200 odds -> payout .5:1
    if pd.isnull(american_odds):
        return None
    if american_odds > 0:
        payout_odds =  american_odds / 100
    elif american_odds < 0:
        payout_odds = 100 / (-1*american_odds)
    return payout_odds

def fractional_odds(payout_odds):
    # win % required to break even
    # payout odds 3:1 -> 25%,  -200 odds -> payout .5:1 -> 66%
    fractional_odds = 1 / (payout_odds + 1)
    return fractional_odds

def spread_bet_odds(row):
    #STDDEV IN PTS DIFFERENCE FROM ALL GAMES
    stddev = 14.525
    avg = row.SCORE_DIFFERENCE_PREDS
    handicap = row.home_handicap

    dist = NormalDist(mu=avg, sigma=stddev)
    return 1 - dist.cdf(0)
    
def joined_odds(
    preds, 
    betting_market, #or moneyline, later player level props
    date_range = ('2023-09-01', '2024-06-01')
):
    # split home and away long ways, returns date, market, name, handicap, payout_odds, predicted_odds
    # convert american odds to fractional odds
    
    preds = pd.read_csv(PREDICTIONS_DIR + preds)
    preds.GAME_DATE = preds.GAME_DATE.apply(lambda d: datetime.strptime(d, '%m/%d/%y').strftime('%Y-%m-%d'))
    odds_df = preds[(preds.GAME_DATE >= date_range[0]) & (preds.GAME_DATE <= date_range[1])]
    
    if betting_market in ('spread', 'moneyline'):
        odds_df['home_handicap'], \
        odds_df['home_odds'], \
        odds_df['away_handicap'], \
        odds_df['away_odds'] = zip(*odds_df.apply(
            partial(opening_line, betting_market=betting_market),
            axis=1
        ))
        odds_df.home_odds = odds_df.home_odds.apply(payout_odds)
        odds_df.away_odds = odds_df.away_odds.apply(payout_odds)
        
        # For players, name should be team name : player name
        # Can expand this logic to maybe approximate % chance from predicted spread
        odds_df['market'] = betting_market
        if betting_market == 'moneyline':
            odds_df['predicted_odds'] = odds_df['GAME_WINNER_PROBS']
        elif betting_market == 'spread':
            odds_df['predicted_odds'] = odds_df.apply(spread_bet_odds, axis=1)
            
        home_odds_df = odds_df[['GAME_DATE', 'market', 'HOME_TEAM_NAME', 'home_handicap', 'home_odds', 'predicted_odds']]
        home_odds_df.columns = ['date', 'market', 'name', 'handicap', 'payout_odds', 'predicted_odds']
        
        odds_df['inverse_odds'] = 1 - odds_df['predicted_odds']
        away_odds_df = odds_df[['GAME_DATE', 'market', 'AWAY_TEAM_NAME', 'away_handicap', 'away_odds', 'inverse_odds']]
        away_odds_df.columns = ['date', 'market', 'name', 'handicap', 'payout_odds', 'predicted_odds']
        
        odds_df_long = pd.concat([home_odds_df, away_odds_df], axis=0).sort_values('date')
        
        return odds_df_long[~pd.isnull(odds_df_long.payout_odds)]


def pick_odds_greater_than(odds, margin=0):
    bets = odds.copy(deep=True)
    bets['breakeven_win_frac'] = bets.payout_odds.apply(fractional_odds) 
    return bets[bets.predicted_odds >= (bets.breakeven_win_frac + margin)]
   
def size_kelly(row):
    b = row.payout_odds
    p = row.predicted_odds
    q = 1 - row.predicted_odds
    return max(0, p - q/b)
    
def kelly_wagers(bets, multiplier=1, cap=1):
    bets['amount'] = bets.apply(size_kelly, axis=1) * multiplier
    wager_history = {
        'bankroll': 10000,
        'betting_units': 'units',
        'wagers': [
            {
                'type': 'single',
                'amount': min(row.amount, cap),
                'bet': {
                    'market': row.market,
                    'date': row.date,
                    'name': row['name'],
                    'handicap': row.handicap,
                    'payout_odds': row.payout_odds
                }
            }
            for i, row in bets.iterrows()
        ]
    }
    return wager_history

def fixed_wagers(bets, amount):
    wager_history = {
        'bankroll': 10000,
        'betting_units': 'dollars',
        'wagers': [
            {
                'type': 'single',
                'amount': amount,
                'bet': {
                    'market': row.market,
                    'date': row.date,
                    'name': row['name'],
                    'handicap': row.handicap,
                    'payout_odds': row.payout_odds
                }
            }
            for i, row in bets.iterrows()
        ]
    }
    return wager_history

def fixed_proportion_wagers(bets, frac):
    wager_history = {
        'bankroll': 10000,
        'betting_units': 'units',
        'wagers': [
            {
                'type': 'single',
                'amount': frac,
                'bet': {
                    'market': row.market,
                    'date': row.date,
                    'name': row['name'],
                    'handicap': row.handicap,
                    'payout_odds': row.payout_odds
                }
            }
            for i, row in bets.iterrows()
        ]
    }
    return wager_history    
    
def evaluation_dataframe(bets, wagers, results):
    tmp = bets.copy(deep=True)
    tmp['betting_unit'] = wagers['betting_units']
    tmp['bet_amount'] = [x['amount'] for x in wagers['wagers']]
    last_row = {col: None for col in bets.columns}
    tmp = pd.concat([tmp, pd.DataFrame(last_row, index=[0])], axis=0, ignore_index=True)
    tmp['bankroll'] = results
    
    tmp['win'] = (tmp.bankroll < tmp.bankroll.shift(-1))
    tmp['win_amount'] = (tmp.bankroll.shift(-1) - tmp.bankroll)
    tmp['other_team'] = tmp.apply(lambda row: get_other_team(row['date'], row['name']), axis=1)
    
    tmp = pd.merge(
        tmp, BOX_SCORES,
        left_on=['date', 'name'],
        right_on=['GAME_DATE', 'TEAM_NAME']
    )
    tmp = pd.merge(
        tmp, BOX_SCORES,
        left_on=['date', 'other_team'],
        right_on=['GAME_DATE', 'TEAM_NAME']
    )

    tmp.rename({'name': 'bet_team', 'PTS_x': 'bet_pts', 'PTS_y': 'other_pts'}, axis=1, inplace=True)
    column_order = ['date', 'bet_team', 'other_team', 'handicap', 'bet_pts', 'other_pts', 'bankroll', 'betting_unit', 'bet_amount', 'win', 'win_amount', 'payout_odds', 'predicted_odds']
    return tmp[column_order]

def get_game(row):
    games = ODDS_GAMES[
        (
            (ODDS_GAMES.away_team==row['name']) | (ODDS_GAMES.home_team==row['name'])
        ) & (ODDS_GAMES.date==row.date)
    ]
    if len(games)==0:
        return None
    return f"{games.iloc[0]['date']} {games.iloc[0]['home_team']} v. {games.iloc[0]['away_team']}"

def pick_random_bets(odds):
    odds = odds.copy(deep=True)
    odds['game'] = odds.apply(get_game, axis=1)
    odds = odds.sample(frac=1)
    odds.drop_duplicates('game', inplace=True)
    return odds
    