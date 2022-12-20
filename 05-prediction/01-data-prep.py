# Author: Morris M. F. Chan
# Date: 2022-12-07

'''
This script takes the raw data for prediction and transform them into the form our models can read.

Usage:
    01-data-prep.py --race_date=<race_date>

Options:
    --race_date=<race_date>     Date of the race.

Example:
    python 05-prediction/01-data-prep.py --race_date='2022-12-07'
'''

import numpy as np
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main( race_date):
    df_race = pd.read_csv( '05-prediction/00-future-input.csv')
    df_horse = pd.read_csv( '01-data/basic-horse.csv') # Retired horse is not needed becuase only current horses are in the race.
    df_jockey = pd.read_csv( '01-data/basic-jockey.csv')
    df_trainer = pd.read_csv( '01-data/basic-trainer.csv')

    df_race_tr = df_race.loc[ :, ~ df_race.columns.str.startswith( 'G_')]

    df_horse = df_horse.iloc[ :, 1:]

    columns_horse = [ 'HorseName', 'HorseOrigin', 'Age', 'YearOfBirth', 'Color', 'Sex', 'H_SeasonalStake', 'H_TotalStake', 'H_1st', 'H_2nd', 'H_3rd', 'H_Total', 'H_Trainer', 'CurrentRating', 'StartofTheSeasonRating']
    df_horse.columns = columns_horse
    df_horse[ 'HorseName'] = df_horse[ 'HorseName'].str.replace( ' ', '', regex = False)
    df_horse[ 'HorseName'] = df_horse[ 'HorseName'].str.replace( r'\(.*\)', '', regex = True)
    df_horse[ 'H_SeasonalStake'] = df_horse[ 'H_SeasonalStake'].str.replace( '$', '', regex = False).str.replace( ',', '').astype( 'int')
    df_horse[ 'H_TotalStake'] = df_horse[ 'H_TotalStake'].str.replace( '$', '', regex = False).str.replace( ',', '').astype( 'int')
    df_horse[ 'H_Rate_win'] = df_horse[ 'H_1st'] / df_horse[ 'H_Total']
    df_horse[ 'H_Rate_place'] = df_horse[ [ 'H_1st', 'H_2nd']].sum( axis = 1) / df_horse[ 'H_Total']
    df_horse[ 'H_Rate_show'] = df_horse[ [ 'H_1st', 'H_2nd', 'H_3rd']].sum( axis = 1) / df_horse[ 'H_Total']

    df_jockey = df_jockey.iloc[ :, 1:]
    columns_jockey = [ 'JockeyName', 'J_1st', 'J_2nd', 'J_3rd', 'J_4th', 'J_5th', 'J_Total', 'J_TotalStakes']
    df_jockey.columns = columns_jockey
    df_jockey[ 'J_ratio_1st'] = df_jockey[ 'J_1st'] / df_jockey[ 'J_Total']
    df_jockey[ 'J_ratio_Top5'] = df_jockey[ [ 'J_1st', 'J_2nd', 'J_3rd', 'J_4th', 'J_5th']].sum( axis = 1) / df_jockey[ 'J_Total']
    df_jockey[ 'J_TotalStakes'] = df_jockey[ 'J_TotalStakes'].str.replace( '$', '', regex = False).str.replace( ',', '').astype( 'int')
    df_jockey[ 'J_Rate_win'] = df_jockey[ 'J_1st'] / df_jockey[ 'J_Total']
    df_jockey[ 'J_Rate_place'] = df_jockey[ ['J_1st', 'J_2nd']].sum( axis = 1) / df_jockey[ 'J_Total']
    df_jockey[ 'J_Rate_show'] = df_jockey[ ['J_1st', 'J_2nd', 'J_3rd']].sum( axis = 1) / df_jockey[ 'J_Total']

    df_trainer = df_trainer.iloc[ :, 1:]
    columns_trainer = [ 'TrainerName', 'T_1st', 'T_2nd', 'T_3rd', 'T_4th', 'T_5th', 'T_Total', 'T_TotalStakes']
    df_trainer.columns = columns_trainer
    df_trainer[ 'T_ratio_1st'] = df_trainer[ 'T_1st'] / df_trainer[ 'T_Total']
    df_trainer[ 'T_ratio_Top5'] = df_trainer[ [ 'T_1st', 'T_2nd', 'T_3rd', 'T_4th', 'T_5th']].sum( axis = 1) / df_trainer[ 'T_Total']
    df_trainer[ 'T_TotalStakes'] = df_trainer[ 'T_TotalStakes'].str.replace( '$', '', regex = False).str.replace( ',', '').astype( 'int')
    df_trainer[ 'T_Rate_win'] = df_trainer[ 'T_1st'] / df_trainer[ 'T_Total']
    df_trainer[ 'T_Rate_place'] = df_trainer[ [ 'T_1st', 'T_2nd']].sum( axis = 1) / df_trainer[ 'T_Total']
    df_trainer[ 'T_Rate_show'] = df_trainer[ [ 'T_1st', 'T_2nd', 'T_3rd']].sum( axis = 1) / df_trainer[ 'T_Total']

    df_merged_j = pd.merge( df_race_tr, df_jockey, left_on = 'Jockey', right_on = 'JockeyName', how = 'left')
    df_merged_jt = pd.merge( df_merged_j, df_trainer, left_on = 'Trainer', right_on = 'TrainerName', how = 'left')
    df_merged_jth = pd.merge( df_merged_jt, df_horse, left_on = 'HorseName', right_on = 'HorseName', how = 'left')
    df_merged_jth[ 'H_Age'] = df_merged_jth[ 'RaceYear'] - df_merged_jth[ 'YearOfBirth']
    df_merged_jth[ 'CurrentRating'] = df_merged_jth[ 'RatingN']

    df_merged_jth[ 'Previous_list'] = df_merged_jth['Previous'].str.split('/')

    df_merged_jth[ 'Previous_list'] = df_merged_jth[ 'Previous_list'].map( to_int)

    df_merged_jth[ 'PreviousPlace'] = df_merged_jth[ 'Previous_list'].map( previous)

    df_merged_jth[ 'AvgPlace3'] = df_merged_jth[ 'Previous_list'].map( avg3)
    df_merged_jth[ 'AvgPlace5'] = df_merged_jth[ 'Previous_list'].map( avg5)

    df_merged_jth[ 'LastRace_nDays'] = pd.Series( dtype = 'float64')
    df_merged_jth[ 'WeightDiff'] = pd.Series( dtype = 'float64')

    for i in range( len( df_merged_jth)):
        try:
            HorseName = df_merged_jth.loc[ i, 'HorseName']
            DeclaredHorseWeight = df_merged_jth.loc[ i, 'DeclaredHorseWeight']
            tmp_table = pd.read_csv( f'01-data/horse-race-history/{HorseName}.csv')
            tmp_table = tmp_table[ ['RDate', 'LastRace_nDays', 'RPlace', 'PreviousPlace', 'AvgPlace3', 'AvgPlace5', 'RRating', 'Weight','WeightDiff']]
            tmp_table[ 'RDate'] = pd.to_datetime( tmp_table[ 'RDate'], format = '%Y-%m-%d')
        except:
            df_merged_jth.loc[ i, 'LastRace_nDays'] = 100
            df_merged_jth.loc[ i, 'WeightDiff'] = 0

        try:
            nDays = (pd.to_datetime( race_date, format = '%Y-%m-%d') - tmp_table.loc[ 0, 'RDate']).days
            df_merged_jth.loc[ i, 'LastRace_nDays'] = nDays
        except:
            df_merged_jth.loc[ i, 'LastRace_nDays'] = 100
        
        try:
            PreviousWeight = tmp_table.loc[ 0, 'Weight']
            WeightDiff = DeclaredHorseWeight - PreviousWeight
            df_merged_jth.loc[ i, 'WeightDiff'] = WeightDiff
        except:
            df_merged_jth.loc[ i, 'WeightDiff'] = 0


    df_merged_jth[ 'CombinedRating'] = df_merged_jth[ 'RatingN']
    df_merged_jth = df_merged_jth.fillna( 0)

    X = df_merged_jth.loc[ :, df_merged_jth.columns.str.startswith( 'ActualWeight') | df_merged_jth.columns.str.startswith( 'DeclaredHorseWeight') | \
        df_merged_jth.columns.str.startswith( 'WinOdds') | df_merged_jth.columns.str.startswith( 'Draw') | \
        df_merged_jth.columns.str.startswith( 'J_Total_') | df_merged_jth.columns.str.startswith( 'J_TotalStakes_') | \
        df_merged_jth.columns.str.startswith( 'J_Rate_') | \
        df_merged_jth.columns.str.startswith( 'T_Total_') | df_merged_jth.columns.str.startswith( 'T_TotalStakes_') | \
        df_merged_jth.columns.str.startswith( 'T_Rate') | \
        df_merged_jth.columns.str.startswith( 'H_Rate_') | df_merged_jth.columns.str.startswith( 'H_Total') | \
        df_merged_jth.columns.str.startswith( 'H_TotalStake') | df_merged_jth.columns.str.startswith( 'CombinedRating') | \
        df_merged_jth.columns.str.startswith( 'H_Age') | df_merged_jth.columns.str.startswith( 'WeightDiff') | \
        df_merged_jth.columns.str.startswith( 'LastRace_nDays') | df_merged_jth.columns.str.startswith( 'PreviousPlace') | \
        df_merged_jth.columns.str.startswith( 'AvgPlace3') | df_merged_jth.columns.str.startswith( 'AvgPlace5') | \
        df_merged_jth.columns.str.startswith( 'Place') | \
        df_merged_jth.columns.str.startswith( 'Race')] # Place for downsampling

    X_mean = X.groupby( 'Race').mean()
    X_std = X.groupby( 'Race').std()

    X_mean.columns = list( map( lambda x: '_'.join( [x, 'mean']), X_mean.columns))
    X_std.columns = list( map( lambda x: '_'.join( [x, 'std']), X_std.columns))

    X_mean_std = pd.concat( [X_mean, X_std], axis = 1)

    X_norm = pd.merge( X, X_mean_std, left_on = 'Race', right_index = True, how = 'inner')
    X_mean_whole = X_norm.loc[:, X_norm.columns.str.endswith( 'mean')]
    X_std_whole = X_norm.loc[:, X_norm.columns.str.endswith( 'std')]
    print( X.shape, X_mean_whole.shape, X_std_whole.shape)

    race_vector = X[ 'Race']
    X = X.drop( 'Race', axis = 1)

    X_norm = pd.DataFrame( ( np.asarray( X) - np.asarray( X_mean_whole)) / np.asarray( X_std_whole))
    X_norm = X_norm.fillna( 0)
    X_norm.columns = list( map( lambda x: '_'.join( [x, 'norm']), X.columns))
    X_comb = pd.concat( [ df_merged_jth, X_norm], axis = 1)

    X_comb.to_csv( '05-prediction/01-preprocessed_20221221.csv', encoding = 'utf-8_sig')


def to_int( x):
    try:
        return list( map( int, x))
    except:
        return [ 10]

def avg3( history):
    try:
        return np.mean( history[ 0:4])
    except:
        return np.mean( history)

def avg5( history):
    try:
        return np.mean( history[ 0:6])
    except:
        return np.mean( history)

def previous( history):
    return history[ 0]

if __name__ == "__main__":
  main( opt[ '--race_date'])