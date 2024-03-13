from random import randint
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
REPO_ROOT = os.environ['REPO_ROOT']
ODDS_DATA = f'{REPO_ROOT}/data/prop_odds_api'
GAME_DATA = f'{REPO_ROOT}/data/original'

games_df = pd.read_csv(f'{ODDS_DATA}/prop_odds_games.csv')
odds_df = pd.read_csv(f'{GAME_DATA}/nba_games_box_scores_2022_2024.csv', usecols=['TEAM_NAME', 'GAME_DATE', 'PTS'])

def get_other_team(date, team_name):
    temp = games_df[
        (
            (games_df.home_team==team_name)
            | (games_df.away_team==team_name)
        ) 
        & (games_df.date==date)
    ]
    if len(temp)!=1:
        print(f'{date} {team_name} {len(temp)} results found')
        
    game = temp.iloc[0]
    
    if game.home_team == team_name:
        return game.away_team
    elif game.away_team == team_name:
        return game.home_team
    
def team_results(date, team_name):
    temp = odds_df[(odds_df.TEAM_NAME==team_name) & (odds_df.GAME_DATE==date)]
    if len(temp)!=1:
        print(f'{date} {team} {len(temp)} results found')
    return temp.iloc[0, -1]
    
def player_results(date, player_name):
    return {
        'points': randint(2,50),
        'assists': randint(2,50),
        'rebounds': randint(2,50)
    }

def bet_wins(bet):
    date = bet['date']
    
    if bet['market'] in ('moneyline', 'spread'):
        # name is team name
        other_team = get_other_team(date, bet['name'])
        return team_results(date, bet['name']) + bet['handicap'] > team_results(date, other_team)
    
    elif bet['market'] == 'player_par':
        # name is team: player
        pass
        
def wager_result(amt, wager):
    #returns net gain/loss from the wager
    # odds are fractional odds
    
    if wager['type']=='single':
        bet = wager['bet']
        if bet_wins(bet):
            return bet['payout_odds'] * amt
        else:
            return -1 * amt
        
    elif wager['type']=='parlay':
        if all([bet_wins(bet) for bet in wager['legs']]):
            outcome = amt
            for bet in wager['legs']:
                outcome += outcome * bet['payout_odds']
            return outcome - amt
        else:
            return -1 * amt
            
        
def simulate_wagers(wager_history):
    current_bankroll = wager_history['bankroll']
    bankroll_history = [current_bankroll]
    
    for wager in wager_history['wagers']:
        if wager_history['betting_units']=='units':
            amt = min(
                current_bankroll * wager['amount'],
                current_bankroll
            )
        elif wager_history['betting_units']=='dollars':
            amt = min(wager['amount'], current_bankroll)
            
        current_bankroll += wager_result(amt, wager)
        bankroll_history.append(current_bankroll)
        
    return bankroll_history

# WAGER HISTORY FORMAT
# wager_history = {
#     'bankroll': 100000,
#     'betting_units': 'units', # or dollars
#     'wagers': [
#         {
#             'type':'parlay', 
#             'amount':50, 
#             'legs':[
#                 {
#                     'market': 'player_par',
#                     'date': '2024-01-23',
#                     'handicap': 48.5,
#                     'payout_odds': 1.1,
#                     'name': 'Over',
#                     'participant_name': 'Nikola Jokic'
#                 },
#                 {
#                     'market': 'spread',
#                     'date': '2024-01-23',
#                     'handicap': -2.5,
#                     'payout_odds': .4,
#                     'team_name': 'Denver Nuggets'
#                 }
#             ]
#         },
#         {
#             'type':'single', 
#             'amount':10, 
#             'bet': {
#                 'market': 'moneyline',
#                 'date': '2024-01-24',
#                 'handicap': 0,
#                 'payout_odds': 1.95,
#                 'team_name': 'Charlotte Hornets'
#             }
#         },
#         {
#             'type':'single', 
#             'amount':10, 
#             'bet': {
#                 'market': 'spread',
#                 'date': '2024-01-24',
#                 'handicap': 8.5,
#                 'payout_odds': .66,
#                 'team_name': 'Washington Wizards'
#             }
#         }
#     ]
# }