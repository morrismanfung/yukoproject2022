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
import os

from functions import *

def main():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']

    pipe_svc_opt = load( '02-model/01-saved-model/02-pipe_svc_opt.joblib')

    svc_dict = load( '02-model/02-saved-scores/02-svc_dict_tmp.joblib')

    svc_dict[ 'test_scores'] = model_testing( pipe_svc_opt, X_train, y_train, X_test, y_test)

    dump( svc_dict, '02-model/02-saved-scores/02-svc_dict.joblib')

def svc_by_proba( pipe_svc, X_test, threshold):
    proba = pipe_svc.decision_function( X_test)
    y_hat = proba > threshold
    return y_hat

def model_testing( pipe_svc_opt, X_train, y_train, X_test, y_test):
    pipe_svc_opt.fit( X_train, y_train)
    y_hat_svc_opt = svc_by_proba( pipe_svc_opt, X_test, 0.40)
    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_svc_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/02-svc_confusion_matrix.csv')
    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_svc_opt, output_dict = True))
    classification_report_.to_csv( '02-model/02-saved-scores/02-svc_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_svc_opt)

if __name__ == '__main__':
    main()