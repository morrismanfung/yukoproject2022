import numpy as np
import pandas as pd

# ---------- Data importation ----------
df_race = pd.read_csv( '01-data/basic-race.csv')
df_horse_current = pd.read_csv( '01-data/basic-horse.csv')
df_horse_retired = pd.read_csv( '01-data/basic-horse-retired.csv')
df_horse = pd.concat( [ df_horse_current, df_horse_retired], axis = 0)
df_jockey = pd.read_csv( '01-data/basic-jockey.csv')
df_trainer = pd.read_csv( '01-data/basic-trainer.csv')

## ---------- Races ----------
columns_race = [
'Place', 'HorseNo', 'HorseName', 'Jockey', 'Trainer', 'ActualWeight', 'DeclaredHorseWeight', 'Draw', '-', 'Position', 'Time', 'WinOdds',
'Date', 'RaceNo', 'Venue', 'Class', 'Distance', 'Going', 'Stake', 'WinnerNo', 'nHorse', 'Winner', 'PLACE'
]

df_race_t = df_race.iloc[ :, 1:] # Sometimes they have 15 horses, which are not used in the model anyway

df_race_t.columns = columns_race

df_race_tr = df_race_t.loc[ :,  ~ df_race_t.columns.str.startswith( '-')]
df_race_tr = df_race_tr.loc[ :,  ~ df_race_tr.columns.str.startswith( 'Position')]
df_race_tr = df_race_tr.loc[ :,  ~ df_race_tr.columns.str.startswith( 'Time')]
# df_race_tr = df_race_tr.loc[ :,  ~ df_race_tr.columns.str.startswith( 'Place')]
df_race_tr = df_race_tr.fillna(0)
df_race_tr = df_race_tr.replace( '---', 0)
df_race_tr[ 'Date'] = pd.to_datetime( df_race_tr[ 'Date'])
df_race_tr[ 'RaceYear'] = pd.DatetimeIndex( df_race_tr[ 'Date']).year
df_race_tr[ 'HorseName'] = df_race_tr[ 'HorseName'].str.replace( r'\(.*\)', '', regex = True)
df_race_tr = df_race_tr.copy()

## ---------- Horses ----------
df_horse = df_horse.iloc[ :, 1:]

columns_horse = [ 'HorseName', 'HorseOrigin', 'Age', 'YearOfBirth', 'Color', 'Sex', 'H_SeasonalStake', 'H_TotalStake', 'H_1st', 'H_2nd', 'H_3rd', 'H_Total', 'H_Trainer', 'CurrentRating', 'StartofTheSeasonRating']
df_horse.columns = columns_horse
df_horse[ 'HorseName'] = df_horse[ 'HorseName'].str.replace( r'\(.*\)', '', regex = True).str.strip()
# df_horse[ 'H_SeasonalStake'] = df_horse[ 'H_SeasonalStake'].str.replace( '$', '').str.replace( ',', '').astype( 'int')
df_horse[ 'H_TotalStake'] = df_horse[ 'H_TotalStake'].str.replace( '$', '').str.replace( ',', '').astype( 'int')
df_horse[ 'H_Rate_win'] = df_horse[ 'H_1st'] / df_horse[ 'H_Total']
df_horse[ 'H_Rate_place'] = df_horse[ [ 'H_1st', 'H_2nd']].sum( axis = 1) / df_horse[ 'H_Total']
df_horse[ 'H_Rate_show'] = df_horse[ [ 'H_1st', 'H_2nd', 'H_3rd']].sum( axis = 1) / df_horse[ 'H_Total']

## ---------- Jockeys ----------
df_jockey = df_jockey.iloc[ :, 1:]
columns_jockey = [ 'JockeyName', 'J_1st', 'J_2nd', 'J_3rd', 'J_4th', 'J_5th', 'J_Total', 'J_TotalStakes']
df_jockey.columns = columns_jockey
df_jockey[ 'J_ratio_1st'] = df_jockey[ 'J_1st'] / df_jockey[ 'J_Total']
df_jockey[ 'J_ratio_Top5'] = df_jockey[ [ 'J_1st', 'J_2nd', 'J_3rd', 'J_4th', 'J_5th']].sum( axis = 1) / df_jockey[ 'J_Total']
df_jockey[ 'J_TotalStakes'] = df_jockey[ 'J_TotalStakes'].str.replace( '$', '').str.replace( ',', '').astype( 'int')
df_jockey[ 'J_Rate_win'] = df_jockey[ 'J_1st'] / df_jockey[ 'J_Total']
df_jockey[ 'J_Rate_place'] = df_jockey[ ['J_1st', 'J_2nd']].sum( axis = 1) / df_jockey[ 'J_Total']
df_jockey[ 'J_Rate_show'] = df_jockey[ ['J_1st', 'J_2nd', 'J_3rd']].sum( axis = 1) / df_jockey[ 'J_Total']

# df_jockey = df_jockey.drop( [ 'J_1st', 'J_2nd', 'J_3rd', 'J_4th', 'J_5th'], axis = 1)
df_jockey.head()

## ---------- Trainers ----------
df_trainer = df_trainer.iloc[ :, 1:]
columns_trainer = [ 'TrainerName', 'T_1st', 'T_2nd', 'T_3rd', 'T_4th', 'T_5th', 'T_Total', 'T_TotalStakes']
df_trainer.columns = columns_trainer
df_trainer[ 'T_ratio_1st'] = df_trainer[ 'T_1st'] / df_trainer[ 'T_Total']
df_trainer[ 'T_ratio_Top5'] = df_trainer[ [ 'T_1st', 'T_2nd', 'T_3rd', 'T_4th', 'T_5th']].sum( axis = 1) / df_trainer[ 'T_Total']
df_trainer[ 'T_TotalStakes'] = df_trainer[ 'T_TotalStakes'].str.replace( '$', '').str.replace( ',', '').astype( 'int')
df_trainer[ 'T_Rate_win'] = df_trainer[ 'T_1st'] / df_trainer[ 'T_Total']
df_trainer[ 'T_Rate_place'] = df_trainer[ [ 'T_1st', 'T_2nd']].sum( axis = 1) / df_trainer[ 'T_Total']
df_trainer[ 'T_Rate_show'] = df_trainer[ [ 'T_1st', 'T_2nd', 'T_3rd']].sum( axis = 1) / df_trainer[ 'T_Total']

# df_trainer = df_trainer.drop( [ 'T_1st', 'T_2nd', 'T_3rd', 'T_4th', 'T_5th'], axis = 1)
df_trainer.head()

## ---------- Merging the information of jockeys ----------
df_merged_j = pd.merge( df_race_tr, df_jockey, left_on = 'Jockey', right_on = 'JockeyName', how = 'left')

## ---------- Merging the information of trainers ----------
df_merged_jt = pd.merge( df_merged_j, df_trainer, left_on = 'Trainer', right_on = 'TrainerName', how = 'left')

## ---------- Merging the information of horses ----------
df_merged_jth = pd.merge( df_merged_jt, df_horse, left_on = 'HorseName', right_on = 'HorseName', how = 'left')
df_merged_jth[ 'H_Age'] = df_merged_jth[ 'RaceYear'] - df_merged_jth[ 'YearOfBirth']

df_merged_jth = df_merged_jth.sort_values( [ 'Date', 'RaceNo'], axis = 0)

## ---------- Extracting the on-race information ----------
df_merged_jth[ 'LastRace_nDays'] = pd.Series()
df_merged_jth[ 'PreviousPlace'] = pd.Series()
df_merged_jth[ 'AvgPlace3'] = pd.Series()
df_merged_jth[ 'AvgPlace5'] = pd.Series()
df_merged_jth[ 'WeightDiff'] = pd.Series()

for i in range( len( df_merged_jth)):
    try:
        HorseName = df_merged_jth.loc[ i, 'HorseName']
        Date = df_merged_jth.loc[ i, 'Date']

        tmp_table = pd.read_csv( f'01-data/02-horses-single-race/{HorseName}.csv')

        tmp_table = tmp_table[ ['RDate', 'LastRace_nDays', 'PreviousPlace', 'AvgPlace3', 'AvgPlace5', 'RRating', 'WeightDiff']]
        tmp_table[ 'RDate'] = pd.to_datetime( tmp_table[ 'RDate'], format = '%Y-%m-%d')

        LastRace_nDays = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 1]
        PreviousPlace = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 2]
        AvgPlace3 = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 3]
        AvgPlace5 = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 4]
        UpdatedRating = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 5]
        WeightDiff = tmp_table[ tmp_table[ 'RDate'] == Date].iloc[ 0, 6]

        df_merged_jth.loc[ i, 'LastRace_nDays'] = LastRace_nDays
        df_merged_jth.loc[ i, 'PreviousPlace'] = PreviousPlace
        df_merged_jth.loc[ i, 'AvgPlace3'] = AvgPlace3
        df_merged_jth.loc[ i, 'AvgPlace5'] = AvgPlace5
        df_merged_jth.loc[ i, 'UpdatedRating'] = UpdatedRating
        df_merged_jth.loc[ i, 'WeightDiff'] = WeightDiff

    except:
        df_merged_jth.loc[ i, 'LastRace_nDays'] = 100
        df_merged_jth.loc[ i, 'PreviousPlace'] = 7
        df_merged_jth.loc[ i, 'AvgPlace3'] = 7
        df_merged_jth.loc[ i, 'AvgPlace5'] = 7
        df_merged_jth.loc[ i, 'UpdatedRating'] = 0
        df_merged_jth.loc[ i, 'WeightDiff'] = 0
    
    print( f'{i} rows updated.')

df_merged_jth[ 'CombinedRating'] = df_merged_jth[ 'CurrentRating'].copy()
for i in range( len( df_merged_jth)):
    if df_merged_jth.loc[ i, 'UpdatedRating'] != 0:
        df_merged_jth.loc[ i, 'CombinedRating'] = df_merged_jth.loc[ i, 'UpdatedRating']

# ---------- Output ----------
df_merged_jth = df_merged_jth.replace( '--', 0)
df_merged_jth = df_merged_jth.fillna( 0)
df_merged_jth.to_csv( '01-data/data-merged_20221218.csv', encoding = 'utf-8_sig')