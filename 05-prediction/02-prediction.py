# Author: Morris M. F. Chan
# 2022-12-19
# Usage: python 05-prediction/02-prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
import pickle

import sys
sys.path.append('05-prediction/')
from classes import *

def main():
    pipe_rfc_opt, pipe_logreg_opt, pipe_lsvc_opt = model_training()
    future_input = pd.read_csv( '05-prediction/01-preprocessed_20221221.csv')
    y_hat_rfc = pipe_rfc_opt.predict( future_input)
    y_hat_logreg = pipe_logreg_opt.predict( future_input)
    y_hat_lsvc = pipe_lsvc_opt.predict( future_input)
    vote = y_hat_rfc.astype( 'int') + y_hat_logreg.astype( 'int') + y_hat_lsvc.astype( 'int')
    y_hat_vote = vote > 1
    y_hat_any = vote > 0

    future_input[ 'rfc'] = y_hat_rfc
    future_input[ 'logreg'] = y_hat_logreg
    future_input[ 'lsvc'] = y_hat_lsvc
    future_input[ 'vote'] = y_hat_vote
    future_input[ 'any'] = y_hat_any
    future_input[ ['Race', 'HorseNo', 'HorseName', 'WinOdds', 'rfc', 'logreg', 'lsvc', 'vote', 'any']][ future_input[ 'any'] == True].to_csv( '05-prediction/03-result.csv', encoding = 'utf-8_sig')
    print( future_input[ ['Race', 'HorseNo', 'HorseName', 'WinOdds', 'rfc', 'logreg', 'lsvc', 'vote', 'any']][ future_input[ 'any'] == True])

def model_training():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']

    column_transformer = load( '02-model/column_transformer.joblib')
    
    with open( '02-model/02-saved-scores/03-rfc_dict.pkl', 'rb') as f:
        rfc_dict = pickle.load( f)
    best_params_rfc = rfc_dict[ 'best_params']
    thld_rfc = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'RFC_pre'])

    with open( '02-model/02-saved-scores/05-logreg_dict.pkl', 'rb') as f:
        logreg_dict = pickle.load( f)
    best_params_logreg = logreg_dict[ 'best_params']
    thld_logreg = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'LogReg_pre'])

    with open( '02-model/02-saved-scores/06-lsvc_dict.pkl', 'rb') as f:
        lsvc_dict = pickle.load( f)
    best_params_lsvc = lsvc_dict[ 'best_params']
    thld_lsvc = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'LinearSVC_pre'])

    pipe_rfc_opt = final_rfc( column_transformer, best_params_rfc, thld_rfc)
    pipe_logreg_opt = final_logreg( column_transformer, best_params_logreg, thld_logreg)
    pipe_lsvc_opt = final_lsvc( column_transformer, best_params_lsvc, thld_lsvc)

    pipe_rfc_opt.fit( X_train, y_train)
    pipe_logreg_opt.fit( X_train, y_train)
    pipe_lsvc_opt.fit( X_train, y_train)

    return pipe_rfc_opt, pipe_logreg_opt, pipe_lsvc_opt

def final_rfc( column_transformer, best_params, thld):
    pipe_rfc_opt = make_pipeline( column_transformer,
                              RFC_thld( n_estimators = best_params[ 'rfc_thld__n_estimators'],
                                        max_features = best_params[ 'rfc_thld__max_features'],
                                        max_depth = best_params[ 'rfc_thld__max_depth'],
                                        criterion = best_params[ 'rfc_thld__criterion'],
                                        bootstrap = best_params[ 'rfc_thld__bootstrap'],
                                        threshold = thld,
                                        random_state = 918))
    return pipe_rfc_opt

def final_logreg( column_transformer, best_params, thld):
    pipe_logreg_opt = make_pipeline( column_transformer,
                                     LogReg_thld( C = best_params[ 'logreg_thld__C'],
                                                  l1_ratio = best_params[ 'logreg_thld__l1_ratio'],
                                                  threshold = thld,
                                                  random_state = 918))
    return pipe_logreg_opt

def final_lsvc( column_transformer, best_params, thld):
    pipe_lsvc_opt = make_pipeline( column_transformer,
                                   LinearSVC_thld( C = best_params[ 'linearsvc_thld__C'],
                                                   threshold = thld,
                                                   random_state = 918))

    return pipe_lsvc_opt

if __name__ == '__main__':
    main()
