# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/07-logreg.py

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
import os

from functions import *

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']
    cv_scoring_metrics = [ 'precision', 'recall', 'f1']

    column_transformer = load( '02-model/column_transformer.joblib')
    pipe_logreg, cv_result_logreg = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_logreg, X_train, y_train)
    pipe_logreg_opt, cv_result_logreg_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    dump( pipe_logreg_opt, '02-model/01-saved-model/05-pipe_logreg_opt.joblib')

    logreg_dict = {
        'cv': cv_result_logreg,
        'best_params': best_params,
        'cv_opt': cv_result_logreg_opt
    }

    dump( logreg_dict, '02-model/02-saved-scores/05-logreg_dict_tmp.joblib')

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_logreg = make_pipeline( column_transformer, LogisticRegression( penalty = 'elasticnet', l1_ratio = 0, max_iter = 2000, tol = 0.01, solver = 'saga', random_state = 918))
    cv_result_logreg = cross_validate( pipe_logreg, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_logreg, cv_result_logreg

def hyperparameter_optimization( pipe_logreg, X_train, y_train):
    param_dist = {
        'logisticregression__C': [ 10**x for x in range( -2, 5)],
        'logisticregression__l1_ratio': [ 0, 0.25, 0.5, 0.75, 1]
    }

    grid_search_logreg = RandomizedSearchCV(
        pipe_logreg, param_dist, cv = 20, scoring = 'precision', n_jobs=-1, return_train_score = True, random_state = 918
    )

    grid_search_logreg.fit( X_train, y_train)
    return grid_search_logreg.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_logreg_opt = make_pipeline( column_transformer,
                                    LogisticRegression( penalty = 'elasticnet', 
                                                        max_iter = 2000,
                                                        tol = 0.01,
                                                        solver = 'saga',
                                                        C = best_params[ 'logisticregression__C'],
                                                        l1_ratio = best_params[ 'logisticregression__l1_ratio'],
                                                        random_state = 918))

    cv_result_logreg_opt = cross_validate( pipe_logreg_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_logreg_opt, cv_result_logreg_opt

def threshold_tuning( pipe_logreg_opt, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_img = pr_curve( pipe_logreg_opt, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    save_chart( pr_curve_img, '02-model/02-saved-scores/05-logreg-pr-purve.png')

if __name__ == '__main__':
    main()