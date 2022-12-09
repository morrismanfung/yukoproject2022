# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/04-svc.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from joblib import dump, load
import pickle
import os

from functions import *

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']
    cv_scoring_metrics = [ 'precision', 'recall', 'f1']

    column_transformer = load( '02-model/column_transformer.joblib')
    pipe_svc, cv_result_svc = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_svc, X_train, y_train)
    pipe_svc_opt, cv_result_svc_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    dump( pipe_svc_opt, '02-model/01-saved-model/02-pipe_svc_opt.joblib')

    svc_dict = {
        'cv': cv_result_svc,
        'best_params': best_params,
        'cv_opt': cv_result_svc_opt
    }

    dump( svc_dict, '02-model/02-saved-scores/02-svc_dict_tmp.joblib')
    with open( '02-model/02-saved-scores/02-svc_dict_tmp.pkl', 'wb') as f:
        pickle.dump( svc_dict, f)
    
    threshold_tuning( pipe_svc_opt, X_train, y_train)

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_svc = make_pipeline( column_transformer, SVC())
    cv_result_svc = cross_validate( pipe_svc, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_svc, cv_result_svc

def hyperparameter_optimization( pipe_svc, X_train, y_train):
    param_dist = {
        'svc__C': [ 10**x for x in range( -2, 5)],
        'svc__gamma': [ 10**x for x in range( -2, 5)]
    }

    random_search_svc = RandomizedSearchCV(
        pipe_svc, param_dist, n_iter = 10, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True, random_state = 918
    )
    random_search_svc.fit( X_train, y_train)
    return random_search_svc.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_svc_opt = make_pipeline( column_transformer,
                              SVC( gamma = best_params[ 'svc__gamma'],
                                   C = best_params[ 'svc__C']))

    cv_result_svc_opt = cross_validate( pipe_svc_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_svc_opt, cv_result_svc_opt

def threshold_tuning( pipe_svc_opt, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_img = pr_curve( pipe_svc_opt, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    pr_curve_img.get_figure().savefig( '02-model/02-saved-scores/02-svc-pr-curve.png')

if __name__ == '__main__':
    main()