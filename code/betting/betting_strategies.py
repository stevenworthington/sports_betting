from datetime import datetime
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from random import randint
from wagers import bet_wins

load_dotenv(find_dotenv())
REPO_ROOT = os.environ['REPO_ROOT']
DATA = f'{REPO_ROOT}/data/prop_odds_api'

moneyline_df = pd.read_csv(f'{DATA}/moneyline.csv')
spread_df = pd.read_csv(f'{DATA}/spread.csv')
games_df = pd.read_csv(f'{DATA}/prop_odds_games.csv')

def casual_fan(start_date, end_date, team_name):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games[(games.away_team==team_name) | (games.home_team==team_name)].sort_values('date')[['game_id', 'date']]

    odds = spread_df[(spread_df.bookie=='betmgm') & (spread_df.name==team_name)]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    odds = odds[~odds.game_id.duplicated()]

    wager_history = {
        'bankroll': 1000,
        'betting_units': 'dollars', # or dollars
        'wagers': [
            {
                'type':'single', 
                'amount':1, 
                'bet': {
                    'market': 'spread',
                    'date': row.date,
                    'handicap': row.handicap,
                    'odds': row.odds,
                    'team_name': row['name']
                }
            }
            for i, row in odds.iterrows()
        ]
    }

    return wager_history

def casual_fan_moneyline(start_date, end_date, team_name):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games[(games.away_team==team_name) | (games.home_team==team_name)].sort_values('date')[['game_id', 'date']]

    odds = moneyline_df[(moneyline_df.bookie=='betmgm') & (moneyline_df.name==team_name)]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    odds = odds[~odds.game_id.duplicated()]

    wager_history = {
        'bankroll': 1000,
        'betting_units': 'dollars', # or dollars
        'wagers': [
            {
                'type':'single', 
                'amount':10, 
                'bet': {
                    'market': 'moneyline',
                    'date': row.date,
                    'handicap': row.handicap,
                    'odds': row.odds,
                    'team_name': row['name']
                }
            }
            for i, row in odds.iterrows()
        ]
    }

    return wager_history

def degen_fan(start_date, end_date, team_name):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games[(games.away_team==team_name) | (games.home_team==team_name)].sort_values('date')[['game_id', 'date']]

    odds = spread_df[(spread_df.bookie=='betmgm') & (spread_df.name==team_name)]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    odds = odds[~odds.game_id.duplicated()]

    wager_history = {
        'bankroll': 1000,
        'betting_units': 'units', # or dollars
        'wagers': [
            {
                'type':'single', 
                'amount':10, 
                'bet': {
                    'market': 'spread',
                    'date': row.date,
                    'handicap': row.handicap,
                    'odds': row.odds,
                    'team_name': row['name']
                }
            }
            for i, row in odds.iterrows()
        ]
    }

    return wager_history

def pick_random(start_date, end_date, units, amount, frac):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games.sort_values('date')[['game_id', 'date', 'home_team', 'away_team']]
    games = games.sample(frac=frac)
    
    odds = spread_df[(spread_df.bookie=='betmgm')]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    odds = odds.groupby('game_id').apply(lambda df: df.sample(1))

    wager_history = {
        'bankroll': 1000,
        'betting_units': units, # or dollars
        'wagers': [
            {
                'type':'single', 
                'amount':amount, 
                'bet': {
                    'market': 'spread',
                    'date': row.date,
                    'handicap': row.handicap,
                    'odds': row.odds,
                    'team_name': row['name']
                }
            }
            for i, row in odds.iterrows()
        ]
    }

    return wager_history

def parlay_demon(start_date, end_date, units, amount):
    #one parlay per day, 2-x
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games.sort_values('date')[['game_id', 'date', 'home_team', 'away_team']]
    
    def pick_parlays(df):
        df = df[~df['name'].duplicated()]
        n = len(df)
        if n < 2:
            return []
        
        legs = randint(2, n)
        df = df.sample(n=max(legs, n))
        
        return [
            {
                'market': 'spread',
                'date': row.date,
                'handicap': row.handicap,
                'odds': row.odds,
                'team_name': row['name']
            }
            for i, row in df.iterrows()
        ]
        
    odds = spread_df[(spread_df.bookie=='betmgm')]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    parlays = odds.groupby('date').apply(pick_parlays).values
    
    wager_history = {
        'bankroll': 1000,
        'betting_units': units,
        'wagers': [
            {
                'type':'parlay', 
                'amount':amount, 
                'legs': parlays[i]
            }
            for i in range(len(parlays))
        ]
    }

    return wager_history

def parlay_god(start_date, end_date, units, amount):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    games = games_df[(games_df.date>=start_date.strftime('%Y-%d-%m')) & (games_df.date<end_date.strftime('%Y-%m-%d'))]
    games = games.sort_values('date')[['game_id', 'date', 'home_team', 'away_team']]
    
    def pick_parlays(df):
        winning_bets = []
        already_bet = []
        
        for i, row in df.iterrows():
            bet = {
                'market': 'spread',
                'date': row.date,
                'handicap': row.handicap,
                'odds': row.odds,
                'team_name': row['name']
            }
            if bet_wins(bet) and not (row['name'] in already_bet):
                winning_bets.append(bet)
                already_bet.append(row['name'])
        return winning_bets

    odds = spread_df[(spread_df.bookie=='betmgm')]
    odds = pd.merge(games, odds, left_on='game_id', right_on='game_id')
    parlays = odds.groupby('date').apply(pick_parlays).values
    
    wager_history = {
        'bankroll': 1000,
        'betting_units': units,
        'wagers': [
            {
                'type':'parlay', 
                'amount':amount, 
                'legs': parlays[i]
            }
            for i in range(len(parlays))
        ]
    }

    return wager_history