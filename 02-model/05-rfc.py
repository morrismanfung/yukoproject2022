# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/05-rfc.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
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
    pipe_rfc = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    best_params = hyperparameter_optimization( pipe_rfc, X_train, y_train)
    # pipe_rfc_opt, cv_result_rfc_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    pipe_rfc_opt = optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics)
    dump( pipe_rfc_opt, '02-model/01-saved-model/03-pipe_rfc_opt.joblib')

    rfc_dict = {
        'best_params': best_params #,
        # 'cv_opt': cv_result_rfc_opt
    }

    dump( rfc_dict, '02-model/02-saved-scores/03-rfc_dict_tmp.joblib')
    with open( '02-model/02-saved-scores/03-rfc_dict_tmp.pkl', 'wb') as f:
        pickle.dump( rfc_dict, f)
    
    threshold_tuning( pipe_rfc_opt, X_train, y_train)

class RFC_thld( RandomForestClassifier):
    def __init__( self, n_estimators = 100, max_depth = None, max_features = 'sqrt', criterion = 'gini',
        bootstrap = True, random_state = None, threshold = None):
        super().__init__( 
            n_estimators = n_estimators,
            max_depth = max_depth,
            max_features = max_features,
            criterion = criterion,
            bootstrap = bootstrap,
            random_state = random_state
        )
        self.threshold = threshold

    def predict( self, X):
        if self.threshold == None:
            predictions = super( RFC_thld, self).predict( X)
        else:
            result = super( RFC_thld, self).predict_proba( X)[ :, 1]
            predictions = np.array( [ True if result >= X.threshold else False])
        return predictions

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_rfc = make_pipeline( column_transformer, RFC_thld( random_state = 918))
    return pipe_rfc

def hyperparameter_optimization( pipe_rfc, X_train, y_train):
    param_dist = {
        'rfc_thld__n_estimators': [ 100*x for x in range( 1, 11)],
        'rfc_thld__max_depth': [ 10*x for x in range( 1, 11)],
        'rfc_thld__max_features': [ 'sqrt', 'log2'],
        'rfc_thld__criterion': [ 'gini', 'entropy', 'log_loss'],
        'rfc_thld__bootstrap': [ True, False]
    }

    random_search_rfc = RandomizedSearchCV(
        pipe_rfc, param_dist, n_iter = 30, cv = 5, scoring = 'precision', n_jobs=-1, return_train_score = True, random_state = 918
    )

    random_search_rfc.fit( X_train, y_train)
    return random_search_rfc.best_params_

def optimized_model( column_transformer, X_train, y_train, best_params, cv_scoring_metrics):
    pipe_rfc_opt = make_pipeline( column_transformer,
                              RFC_thld( n_estimators = best_params[ 'rfc_thld__n_estimators'],
                                        max_features = best_params[ 'rfc_thld__max_features'],
                                        max_depth = best_params[ 'rfc_thld__max_depth'],
                                        criterion = best_params[ 'rfc_thld__criterion'],
                                        bootstrap = best_params[ 'rfc_thld__bootstrap']))

    # cv_result_rfc_opt = cross_validate( pipe_rfc_opt, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_rfc_opt #, cv_result_rfc_opt

def threshold_tuning( pipe_rfc_opt, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_df, pr_curve_img = pr_curve( pipe_rfc_opt, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    pr_curve_df.to_csv( '02-model/02-saved-scores/03-rfc-thresholds.csv')
    pr_curve_img.get_figure().savefig( '02-model/02-saved-scores/03-rfc-pr-curve.png')

if __name__ == '__main__':
    main()