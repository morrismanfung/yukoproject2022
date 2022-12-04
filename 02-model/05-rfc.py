# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/05-rfc.py

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
    pipe_rfc, cv_result_rfc = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_rfc, X_train, y_train)
    pipe_rfc_opt, cv_result_rfc_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    dump( pipe_rfc_opt, '02-model/01-saved-model/03-pipe_rfc_opt.joblib')

    rfc_dict = {
        'cv': cv_result_rfc,
        'best_params': best_params,
        'cv_opt': cv_result_rfc_opt
    }

    dump( rfc_dict, '02-model/02-saved-scores/03-rfc_dict_tmp.joblib')

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_rfc = make_pipeline( column_transformer, RandomForestClassifier( random_state = 918))
    cv_result_rfc = cross_validate( pipe_rfc, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_rfc, cv_result_rfc

def hyperparameter_optimization( pipe_rfc, X_train, y_train):
    param_dist = {
        'randomforestclassifier__n_estimators': [ 100*x for x in range( 1, 11)],
        'randomforestclassifier__max_depth': [ 10*x for x in range( 1, 11)],
        'randomforestclassifier__max_features': [ 'sqrt', 'log2'],
        'randomforestclassifier__criterion': [ 'gini', 'entropy', 'log_loss'],
        'randomforestclassifier__bootstrap': [ True, False]
    }

    random_search_rfc = RandomizedSearchCV(
        pipe_rfc, param_dist, n_iter = 30, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True, random_state = 918
    )

    random_search_rfc.fit( X_train, y_train)
    return random_search_rfc.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_rfc_opt = make_pipeline( column_transformer,
                              RandomForestClassifier( n_estimators = best_params[ 'randomforestclassifier__n_estimators'],
                                                      max_features = best_params[ 'randomforestclassifier__max_features'],
                                                      max_depth = best_params[ 'randomforestclassifier__max_depth'],
                                                      criterion = best_params[ 'randomforestclassifier__criterion'],
                                                      bootstrap = best_params[ 'randomforestclassifier__bootstrap']))

    cv_result_rfc_opt = cross_validate( pipe_rfc_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_rfc_opt, cv_result_rfc_opt

def threshold_tuning( pipe_rfc_opt, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_img = pr_curve( pipe_rfc_opt, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    save_chart( pr_curve_img, '02-model/02-saved-scores/03-rfc-pr-purve.png')

if __name__ == '__main__':
    main()