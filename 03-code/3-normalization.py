import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_full = pd.read_csv( '..\\01-data\\data-merged_20220910.csv')
data_full = data_full[ data_full[ 'H_Age'] < 100] # To filter out any problematic entries.
# data_full = data_full[ data_full[ 'Place'] < 11]
# data_full = data_full[ data_full[ 'RaceYear'] > 2020]

data_full[ 'RaceID'] = data_full[ 'Date'].str.cat( list( data_full[ 'RaceNo'].astype( str)), sep = '-')

X = data_full.loc[ :, data_full.columns.str.startswith( 'ActualWeight') | data_full.columns.str.startswith( 'DeclaredHorseWeight') | \
    data_full.columns.str.startswith( 'WinOdds') | data_full.columns.str.startswith( 'Draw') | \
    data_full.columns.str.startswith( 'J_Total_') | data_full.columns.str.startswith( 'J_TotalStakes_') | \
    data_full.columns.str.startswith( 'J_Rate_') | \
    data_full.columns.str.startswith( 'T_Total_') | data_full.columns.str.startswith( 'T_TotalStakes_') | \
    data_full.columns.str.startswith( 'T_Rate') | \
    data_full.columns.str.startswith( 'H_Rate_') | data_full.columns.str.startswith( 'H_Total') | \
    data_full.columns.str.startswith( 'H_TotalStake') | data_full.columns.str.startswith( 'CombinedRating') | \
    data_full.columns.str.startswith( 'H_Age') | data_full.columns.str.startswith( 'WeightDiff') | \
    data_full.columns.str.startswith( 'LastRace_nDays') | data_full.columns.str.startswith( 'PreviousPlace') | \
    data_full.columns.str.startswith( 'AvgPlace3') | data_full.columns.str.startswith( 'AvgPlace5') | \
    data_full.columns.str.startswith( 'Place') | \
    data_full.columns.str.startswith( 'RaceID')] # Place for downsampling

y = data_full[ 'Winner']

X_mean = X.groupby( 'RaceID').mean()
X_std = X.groupby( 'RaceID').std()

X_mean.columns = list( map( lambda x: '_'.join( [x, 'mean']), X_mean.columns))
X_std.columns = list( map( lambda x: '_'.join( [x, 'std']), X_std.columns))

X_mean_std = pd.concat( [X_mean, X_std], axis = 1)

X_norm = pd.merge( X, X_mean_std, left_on = 'RaceID', right_index = True, how = 'inner')

X_mean_whole = X_norm.loc[:, X_norm.columns.str.endswith( 'mean')]
X_std_whole = X_norm.loc[:, X_norm.columns.str.endswith( 'std')]

X_norm = pd.DataFrame( (np.asarray( X.drop( 'RaceID', axis = 1)) - np.asarray( X_mean_whole)) / np.asarray( X_std_whole))
X_norm  = X_norm.fillna( 0)
X_norm.columns = list( map( lambda x: '_'.join( [x, 'norm']), X.drop( 'RaceID', axis = 1).columns))

X_comb = pd.concat( [ data_full, X_norm], axis = 1)

X_comb.to_csv( '..\\01-data\\01-historical\\data-merged_20220910_norm.csv', encoding = 'utf-8_sig')