import pandas as pd
from functools import partial

PREDICTIONS_DIR = '../../data/model_predictions/'
ODDS_DIR = '../../data/prop_odds_api/'
ODDS = {
    'moneyline': pd.read_csv(ODDS_DIR + 'moneyline.csv'),
    'spread': pd.read_csv(ODDS_DIR + 'spread.csv')
}
ODDS_GAMES = pd.read_csv(ODDS_DIR + 'prop_odds_games.csv')


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

def joined_odds(
    preds, 
    betting_market, #or moneyline, later player level props
    date_range = ('2023-09-01', '2024-06-01')
):
    # split home and away long ways, returns date, market, name, handicap, payout_odds, predicted_odds
    # convert american odds to fractional odds
    
    preds = pd.read_csv(PREDICTIONS_DIR + preds)
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
        home_odds_df = odds_df[['GAME_DATE', 'market', 'HOME_TEAM_NAME', 'home_handicap', 'home_odds', 'GAME_RESULT_PREDS']]
        home_odds_df.columns = ['date', 'market', 'name', 'handicap', 'payout_odds', 'predicted_odds']

        odds_df['INVERSE_GAME_RESULT_PREDS'] = 1 - odds_df['GAME_RESULT_PREDS']
        away_odds_df = odds_df[['GAME_DATE', 'market', 'AWAY_TEAM_NAME', 'away_handicap', 'away_odds', 'INVERSE_GAME_RESULT_PREDS']]
        away_odds_df.columns = ['date', 'market', 'name', 'handicap', 'payout_odds', 'predicted_odds']

        odds_df_long = pd.concat([home_odds_df, away_odds_df], axis=0).sort_values('date')
        
        return odds_df_long[~pd.isnull(odds_df_long.payout_odds)]


def pick_moneyline_greater_than(odds):
    bets = odds.copy(deep=True)
    bets['breakeven_win_frac'] = bets.payout_odds.apply(fractional_odds) 
    return bets[bets.predicted_odds >= bets.breakeven_win_frac]
    
def kelly_wagers(bets):
    def size_kelly(row):
        b = row.payout_odds
        p = row.predicted_odds
        q = 1 - row.predicted_odds
        return p - q/b
    
    bets['amount'] = bets.apply(size_kelly, axis=1)
    wager_history = {
        'bankroll': 1000,
        'betting_units': 'units',
        'wagers': [
            {
                'type': 'single',
                'amount': row.amount,
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