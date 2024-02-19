from random import randint
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
REPO_ROOT = os.environ['REPO_ROOT']
ODDS_DATA = f'{REPO_ROOT}/data/prop_odds_api'
GAME_DATA = f'{REPO_ROOT}/data/original'

games_df = pd.read_csv(f'{ODDS_DATA}/prop_odds_games.csv')
odds_df = pd.read_csv(f'{GAME_DATA}/nba_games_box_scores_2015_2024.csv', usecols=['TEAM_NAME', 'GAME_DATE', 'PTS'])

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
        other_team = get_other_team(date, bet['team_name'])
        return team_results(date, bet['team_name']) + bet['handicap'] > team_results(date, other_team)
    
    elif bet['market'] == 'player_par':
        res = player_results(date, bet['participant_name'])
        if 'over' in bet['name'].lower():
            return res['points'] + res['assists'] + res['rebounds'] > bet['handicap']                    
        elif 'under' in bet['name'].lower():
            return res['points'] + res['assists'] + res['rebounds'] < bet['handicap'] 
        
def wager_result(amt, wager):
    if wager['type']=='single':
        bet = wager['bet']
        if bet_wins(bet):
            if bet['odds'] > 0:
                return amt * bet['odds'] / 100
            elif bet['odds'] < 0:
                return amt / (-1*bet['odds']) * 100
        else:
            return -1 * amt
        
    elif wager['type']=='parlay':
        if all([bet_wins(bet) for bet in wager['legs']]):
            outcome = amt
            for bet in wager['legs']:
                if bet['odds'] > 0:
                    outcome += outcome * bet['odds'] / 100
                elif bet['odds'] < 0:
                    outcome += outcome / (-1*bet['odds']) * 100
            return outcome - amt
        else:
            return -1 * amt
            
        
def simulate_bets(wager_history):
    current_bankroll = wager_history['bankroll']
    bankroll_history = [current_bankroll]
    
    for wager in wager_history['wagers']:
        if wager_history['betting_units']=='units':
            amt = min(
                current_bankroll * wager['amount'] / 100,
                current_bankroll
            )
        elif wager_history['betting_units']=='dollars':
            amt = min(wager['amount'], current_bankroll)
            
        current_bankroll += wager_result(amt, wager)
        bankroll_history.append(current_bankroll)
        
    return bankroll_history