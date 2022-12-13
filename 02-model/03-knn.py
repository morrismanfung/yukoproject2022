# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/03-knn.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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
    pipe_knn = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_knn, X_train, y_train)
    pipe_knn_opt, cv_result_knn_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    try:
        dump( pipe_knn_opt, '02-model/01-saved-model/01-pipe_knn_opt.joblib')
    except:
        os.makedirs( os.path.dirname( '02-model/01-saved-model/'))
        dump( pipe_knn_opt, '02-model/01-saved-model/01-pipe_knn_opt.joblib')

    test_scores = model_testing( pipe_knn_opt, X_train, y_train, X_test, y_test)

    knn_dict = {
        'best_params': best_params,
        'cv_scores': cv_result_knn_opt,
        'test_scores': test_scores
    }

    try:
        dump( knn_dict, '02-model/02-saved-scores/01-knn_dict.joblib')
    except:
        os.makedirs( os.path.dirname( '02-model/02-saved-scores/'))
        dump( knn_dict, '02-model/02-saved-scores/01-knn_dict.joblib')
    with open( '02-model/02-saved-scores/01-knn_dict.pkl', 'wb') as f:
        pickle.dump( knn_dict, f)
    
    with open( 'bin/03-knn', 'w') as f:
        f.close()


def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_knn = make_pipeline( column_transformer, KNeighborsClassifier())
    return pipe_knn

def hyperparameter_optimization( pipe_knn, X_train, y_train):
    param_grid = {
        "kneighborsclassifier__n_neighbors": list( range( 5, 35, 5))
    }

    grid_search_knn = GridSearchCV(
        pipe_knn, param_grid, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True
    )
    grid_search_knn.fit( X_train, y_train)
    return grid_search_knn.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_knn_opt = make_pipeline( column_transformer,
                              KNeighborsClassifier( n_neighbors = best_params[ 'kneighborsclassifier__n_neighbors']))
    cv_result_knn_opt = cross_validate( pipe_knn_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_knn_opt, cv_result_knn_opt

def model_testing( pipe_knn_opt, X_train, y_train, X_test, y_test):
    pipe_knn_opt.fit( X_train, y_train)
    y_hat_knn_opt = pipe_knn_opt.predict( X_test)
    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_knn_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/01-knn_confusion_matrix.csv')

    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_knn_opt, output_dict = True))
    try:
        classification_report_.to_csv( '02-model/02-saved-scores/01-knn_classification_report.csv')
    except:
        os.makedirs( os.path.dirname( '02-model/02-saved-scores/'))
        classification_report_.to_csv( '02-model/02-saved-scores/01-knn_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_knn_opt)

if __name__ == '__main__':
    main()