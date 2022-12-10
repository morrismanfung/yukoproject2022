# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/08-lsvc.py

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
    pipe_lsvc = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_lsvc, X_train, y_train)
    pipe_lsvc_opt, cv_result_lsvc_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    dump( pipe_lsvc_opt, '02-model/01-saved-model/06-pipe_lsvc_opt.joblib')

    lsvc_dict = {
        'best_params': best_params,
        'cv_opt': cv_result_lsvc_opt
    }

    dump( lsvc_dict, '02-model/02-saved-scores/06-lsvc_dict_tmp.joblib')
    with open( '02-model/02-saved-scores/06-lsvc_dict_tmp.pkl', 'wb') as f:
        pickle.dump( lsvc_dict, f)
    
    threshold_tuning( pipe_lsvc_opt, X_train, y_train)

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_lsvc = make_pipeline( column_transformer, LinearSVC( dual = False, random_state = 918))
    return pipe_lsvc

def hyperparameter_optimization( pipe_lsvc, X_train, y_train):
    param_dist = {
        'linearsvc__C': [ 10**x for x in range( -2, 5)]
    }

    grid_search_lsvc = GridSearchCV(
        pipe_lsvc, param_dist, cv = 20, scoring = 'precision', n_jobs=-1, return_train_score = True
    )

    grid_search_lsvc.fit( X_train, y_train)
    return grid_search_lsvc.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_lsvc_opt = make_pipeline( column_transformer,
                                    LinearSVC( dual = False,
                                                C = best_params[ 'linearsvc__C'],
                                                random_state = 918))

    cv_result_lsvc_opt = cross_validate( pipe_lsvc_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_lsvc_opt, cv_result_lsvc_opt

def threshold_tuning( pipe_lsvc_opt, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_img = pr_curve( pipe_lsvc_opt, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    pr_curve_img.get_figure().savefig( '02-saved-scores/06-lsvc-pr-curve.png')

if __name__ == '__main__':
    main()