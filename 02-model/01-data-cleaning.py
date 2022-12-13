
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from functions import *

def main():
    data_full = pd.read_csv( '01-data/data_merged_20220910_norm.csv')
    data_full = data_full.query('H_Age<100&Draw>0').reset_index( drop = True)
    y = data_full[ 'Winner']

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
        data_full.columns.str.startswith( 'Place')] # Place is added for downsampling

    data_train, data_test = train_test_split( pd.concat( [X, y], axis = 1), test_size = 0.2, stratify = y, random_state = 918)
    data_train.to_csv( '01-data/train.csv', index = False)
    data_test.to_csv( '01-data/test.csv', index = False)

    try:
        with open( 'bin/01-cleaned-data', 'w') as f:
            f.close()
    except:
        os.makedirs( os.path.dirname( 'bin/'))
        with open( 'bin/01-cleaned-data', 'w') as f:
            f.close()

if __name__ == '__main__':
    main()

    

