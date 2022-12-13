# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/02-column-transformer.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from joblib import dump

from functions import *

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']
    scoring_metrics = [ 'precision', 'recall', 'f1']

    cols = {
        'cols_std':[ 'ActualWeight', 'DeclaredHorseWeight', 'Draw', 'H_Age', 'PreviousPlace', 'AvgPlace3', 'AvgPlace5', 'WeightDiff', 'CombinedRating'],
        'cols_log_std': [ 'WinOdds', 'H_TotalStake', 'H_Total', 'LastRace_nDays'],
        'cols_passthrough': [ 'J_Rate_win', 'J_Rate_place', 'J_Rate_show', 'T_Rate_win', 'T_Rate_place', 'T_Rate_show',
            'H_Rate_win', 'H_Rate_place', 'H_Rate_show',
            'ActualWeight_norm', 'DeclaredHorseWeight_norm', 'WinOdds_norm',
            'J_Rate_win_norm', 'J_Rate_place_norm', 'J_Rate_show_norm', 'T_Rate_win_norm', 'T_Rate_place_norm', 'T_Rate_show_norm',
            'H_TotalStake_norm', 'H_Total_norm', 'H_Rate_win_norm', 'H_Rate_place_norm', 'H_Rate_show_norm', 'H_Age_norm',
            'LastRace_nDays_norm', 'PreviousPlace_norm', 'AvgPlace3_norm', 'AvgPlace5_norm', 'WeightDiff_norm', 'CombinedRating_norm'],
        'cols_drop': ['Place', 'Place_norm', 'Draw_norm']
    }

    X_train_transformed = transform_data_pre_selection( X_train, cols)
    column_transformer = column_transformer_post_selection( X_train_transformed, y_train, cols)
    dump( column_transformer, '02-model/column_transformer.joblib')

    with open( 'bin/02-column-transformer', 'w') as f:
        f.close()

def transform_data_pre_selection( X_train, cols):
    log_transformer = FunctionTransformer( log_func)
    pipe_log_std = make_pipeline(
        log_transformer, StandardScaler()
    )

    column_transformer_pre = make_column_transformer(
        ( StandardScaler(), cols[ 'cols_std']),
        ( pipe_log_std, cols[ 'cols_log_std']),
        ( 'passthrough', cols['cols_passthrough']),
        ( 'drop', cols['cols_drop'])
    )
    
    return pd.DataFrame( column_transformer_pre.fit_transform( X_train),
        columns = cols['cols_std'] + cols['cols_log_std'] + cols['cols_passthrough'])

def if_in( raw, reference):
    '''
    Return a list of elements in raw if they also exist in reference.
    '''
    return list( filter( lambda x: True if x in reference else False, raw))

def column_transformer_post_selection( X_train_transformed, y_train, cols):
    rfecv = RFE( LogisticRegression( max_iter = 10000, solver = 'saga', random_state = 918), n_features_to_select = 0.75)
    rfecv.fit( X_train_transformed, y_train)
    print( len( rfecv.feature_names_in_[ rfecv.support_]))
    print( rfecv.feature_names_in_[ rfecv.support_])
    features_df = pd.DataFrame(
        {
            'feature': rfecv.feature_names_in_[ rfecv.support_]
        }
    )
    features_df.to_csv( '02-model/features_selected.csv')
    cols_std = if_in( cols[ 'cols_std'], rfecv.feature_names_in_[ rfecv.support_])
    cols_log_std = if_in( cols[ 'cols_log_std'], rfecv.feature_names_in_[ rfecv.support_])
    cols_passthrough = if_in( cols[ 'cols_passthrough'], rfecv.feature_names_in_[ rfecv.support_])
    cols_drop = list( filter( lambda x: False if x in rfecv.feature_names_in_[ rfecv.support_] else True, X_train_transformed.columns))

    log_transformer = FunctionTransformer( log_func)
    pipe_log_std = make_pipeline(
        log_transformer, StandardScaler()
    )
    column_transformer = make_column_transformer(
        ( StandardScaler(), cols_std),
        ( pipe_log_std, cols_log_std),
        ( 'passthrough', cols_passthrough),
        ( 'drop', cols_drop)
    )
    return column_transformer

if __name__ == '__main__':
    main()