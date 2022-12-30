# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/06-lsvc-test.py

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
from classes import LinearSVC_thld

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']
    cv_scoring_metrics = [ 'precision', 'recall', 'f1']
    
    column_transformer = load( '02-model/column_transformer.joblib')
    
    with open( '02-model/02-saved-scores/06-lsvc_dict_tmp.pkl', 'rb') as f:
        lsvc_dict = pickle.load( f)
    best_params = lsvc_dict[ 'best_params']

    thld = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'LinearSVC_pre'])
    
    pipe_lsvc_opt = final_lsvc( column_transformer, best_params, thld)
    lsvc_dict[ 'cv_scores'] = cross_validate( pipe_lsvc_opt, X_train, y_train, cv = 5, scoring = cv_scoring_metrics, return_train_score = True)
    lsvc_dict[ 'test_scores'] = model_testing( pipe_lsvc_opt, X_train, y_train, X_test, y_test)

    dump( lsvc_dict, '02-model/02-saved-scores/06-lsvc_dict.joblib')
    with open( '02-model/02-saved-scores/06-lsvc_dict.pkl', 'wb') as f:
        pickle.dump( lsvc_dict, f)
    
    with open( 'bin/08-lsvc-test', 'w') as f:
        f.close()

def final_lsvc( column_transformer, best_params, thld):
    pipe_lsvc_opt = make_pipeline( column_transformer,
                                   LinearSVC_thld( C = best_params[ 'linearsvc_thld__C'],
                                                   threshold = thld,
                                                   random_state = 918))

    return pipe_lsvc_opt

def model_testing( pipe_lsvc_opt, X_train, y_train, X_test, y_test):
    pipe_lsvc_opt.fit( X_train, y_train)
    y_hat_lsvc_opt = pipe_lsvc_opt.predict( X_test)
    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_lsvc_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/06-lsvc_confusion_matrix.csv')
    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_lsvc_opt, output_dict = True))
    classification_report_.to_csv( '02-model/02-saved-scores/06-lsvc_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_lsvc_opt, X_test)

if __name__ == '__main__':
    main()