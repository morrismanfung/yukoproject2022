# Author: Morris M. F. Chan
# 2022-12-02

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import altair as alt
alt.renderers.enable('mimetype')
alt.data_transformers.enable('data_server')

from functions import *

def main():
    data_full = pd.read_csv( '../01-data/data_merged_20220910_norm.csv')
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

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, stratify = y, random_state = 918)

    scoring_metrics = [ 'precision', 'recall', 'f1']

    cols = columns_types()

    X_train_transformed = column_transformer_pre_selection( X_train)

def columns_types():

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

    return cols

def column_transformer_pre_selection( X_train, cols):
    
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
    
    return pd.DataFrame( column_transformer_pre.fit_transform( X_train), columns = cols_std + cols_log_std + cols_passthrough)

def column_transformer_post_selection( X_train_transformed, y_train):
    rfecv = RFE( LogisticRegression( max_iter = 1000, solver = 'saga'), n_features_to_select = 0.75)
    rfecv.fit( X_train_transformed, y_train)
    cols_std = selected_feat( cols_std)


