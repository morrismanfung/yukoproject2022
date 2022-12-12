# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/04-svc-test.py

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
from classes import SVC_thld

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']
    column_transformer = load( '02-model/column_transformer.joblib')

    with open( '02-model/02-saved-scores/02-svc_dict_tmp.pkl', 'rb') as f:
        svc_dict = pickle.load( f)
    best_params = svc_dict[ 'best_params']

    thld = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'SVC'])

    pipe_svc_opt = final_svc( column_transformer, best_params, thld)
    svc_dict[ 'test_scores'] = model_testing( pipe_svc_opt, X_train, y_train, X_test, y_test, thld)

    dump( svc_dict, '02-model/02-saved-scores/02-svc_dict.joblib')
    with open( '02-model/02-saved-scores/02-svc_dict.pkl', 'wb') as f:
        pickle.dump( svc_dict, f)

def final_svc( column_transformer, best_params, thld):
    pipe_svc_opt = make_pipeline( column_transformer,
                              SVC_thld( gamma = best_params[ 'svc_thld__gamma'],
                                   C = best_params[ 'svc_thld__C'],
                                   threshold = thld,
                                   random_state = 918))

    return pipe_svc_opt

def model_testing( pipe_svc_opt, X_train, y_train, X_test, y_test, thld):
    pipe_svc_opt.fit( X_train, y_train)
    y_hat_svc_opt = pipe_svc_opt.predict( X_test)

    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_svc_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/02-svc_confusion_matrix.csv')

    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_svc_opt, output_dict = True))
    classification_report_.to_csv( '02-model/02-saved-scores/02-svc_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_svc_opt)

if __name__ == '__main__':
    main()