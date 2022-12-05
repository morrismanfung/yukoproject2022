# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/05-rfc-test.py

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

    pipe_rfc_opt = load( '02-model/01-saved-model/03-pipe_rfc_opt.joblib')

    with open( '02-model/02-saved-scores/03-rfc_dict_tmp.pkl', 'rb') as f:
        rfc_dict = pickle.load( f)

    thld = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'RFC'])
    rfc_dict[ 'test_scores'] = model_testing( pipe_rfc_opt, X_train, y_train, X_test, y_test, thld)

    dump( rfc_dict, '02-model/02-saved-scores/03-rfc_dict.joblib')
    with open( '02-model/02-saved-scores/03-rfc_dict.pkl', 'wb') as f:
        pickle.dump( rfc_dict, f)

def rfc_by_proba( pipe_rfc, X_test, threshold):
    proba = pipe_rfc.predict_proba( X_test)[ :, 1]
    y_hat = proba > threshold
    return y_hat

def model_testing( pipe_rfc_opt, X_train, y_train, X_test, y_test, thld):
    pipe_rfc_opt.fit( X_train, y_train)
    y_hat_rfc_opt = rfc_by_proba( pipe_rfc_opt, X_test, thld)
    confusion_matrix_ = better_confusion_matrix( y_test, y_hat_rfc_opt, labels = [ True, False])
    confusion_matrix_.to_csv( '02-model/02-saved-scores/03-rfc_confusion_matrix.csv')
    classification_report_ = pd.DataFrame( classification_report( y_test, y_hat_rfc_opt, output_dict = True))
    classification_report_.to_csv( '02-model/02-saved-scores/03-rfc_classification_report.csv')
    return test_scoring_metrics( y_test, y_hat_rfc_opt)

if __name__ == '__main__':
    main()