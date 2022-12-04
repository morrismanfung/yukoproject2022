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
import os

from functions import *

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']

    pipe_lsvc_opt = load( '02-model/01-saved-model/06-pipe_lsvc_opt.joblib')

    lsvc_dict = load( '02-model/02-saved-scores/06-lsvc_dict_tmp.joblib')

    lsvc_dict[ 'test_scores'] = model_testing( pipe_lsvc_opt, X_train, y_train, X_test, y_test)

    dump( lsvc_dict, '02-model/02-saved-scores/06-lsvc_dict.joblib')

def lsvc_by_proba( pipe_lsvc, X_test, threshold):
    proba = pipe_lsvc.decision_function( X_test)
    y_hat = proba > threshold
    return y_hat

def model_testing( pipe_lsvc_opt, X_train, y_train, X_test, y_test):
    pipe_lsvc_opt.fit( X_train, y_train)
    y_hat_lsvc_opt = lsvc_by_proba( pipe_lsvc_opt, X_test, -0.0113)
    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_lsvc_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/06-lsvc_confusion_matrix.csv')
    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_lsvc_opt, output_dict = True))
    classification_report_.to_csv( '02-model/02-saved-scores/06-lsvc_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_lsvc_opt)

if __name__ == '__main__':
    main()