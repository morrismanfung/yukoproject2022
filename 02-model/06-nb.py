# Author: Morris M. F. Chan
# 2022-12-02
# Usage: python 02-model/06-nb.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
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
    pipe_nb, cv_result_nb = basic_model( column_transformer, X_train, y_train, cv_scoring_metrics)
    dump( pipe_nb, '02-model/01-saved-model/04-pipe_nb.joblib')

    nb_dict = {
        'best_params': 'None',
        'cv_opt': cv_result_nb
    }

    dump( nb_dict, '02-model/02-saved-scores/04-nb_dict_tmp.joblib')
    with open( '02-model/02-saved-scores/04-nb_dict_tmp.pkl', 'wb') as f:
        pickle.dump( nb_dict, f)
    
    threshold_tuning( pipe_nb, X_train, y_train)

def basic_model( column_transformer, X_train, y_train, cv_scoring_metrics):
    pipe_nb = make_pipeline( column_transformer, PowerTransformer(), GaussianNB())
    cv_result_nb = cross_validate( pipe_nb, X_train, y_train, cv = 5, return_train_score = True, scoring = cv_scoring_metrics)
    return pipe_nb, cv_result_nb

def threshold_tuning( pipe_nb, X_train, y_train):
    X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 918)
    pr_curve_img = pr_curve( pipe_nb, X_cv_train, X_cv_test, y_cv_train, y_cv_test)
    pr_curve_img.get_figure().savefig( '02-model/02-saved-scores/04-nb-pr-curve.png')

if __name__ == '__main__':
    main()