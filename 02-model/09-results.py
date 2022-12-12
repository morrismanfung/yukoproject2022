# Author: Morris M. F. Chan
# 2022-12-02

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
from joblib import dump, load
import pickle

def main():
    knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict = read_results()
    cv_df = cv_results( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)
    cv_df.to_csv( '02-model/02-saved-scores/07-cross-validation-results.csv')
    print_best_parameters( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)
    test_scores_df = test_scores( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)
    test_scores_df.to_csv( '02-model/02-saved-scores/08-test-results.csv')

def read_results():
    with open( '02-model/02-saved-scores/01-knn_dict.pkl', 'rb') as f:
        knn_dict = pickle.load( f)
    with open( '02-model/02-saved-scores/02-svc_dict.pkl', 'rb') as f:
        svc_dict = pickle.load( f)
    with open( '02-model/02-saved-scores/03-rfc_dict.pkl', 'rb') as f:
        rfc_dict = pickle.load( f)
    with open( '02-model/02-saved-scores/04-nb_dict.pkl', 'rb') as f:
        nb_dict = pickle.load( f)
    with open( '02-model/02-saved-scores/05-logreg_dict.pkl', 'rb') as f:
        logreg_dict = pickle.load( f)
    with open( '02-model/02-saved-scores/06-lsvc_dict.pkl', 'rb') as f:
        lsvc_dict = pickle.load( f)
    
    return knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict

def cv_results( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict):
    cv_dict = {}
    #cv_dict[ 'KNN'] = pd.DataFrame( knn_dict[ 'cv']).T.agg( [ 'mean', 'std'], axis = 1)
    #cv_dict[ 'KNN_opt'] = pd.DataFrame( knn_dict[ 'cv_opt']).T.agg( [ 'mean', 'std'], axis = 1)
    cv_dict[ 'SVC'] = pd.DataFrame( svc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'RFC'] = pd.DataFrame( rfc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'NB'] = pd.DataFrame( nb_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'LogReg'] = pd.DataFrame( logreg_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'LinearSVC'] = pd.DataFrame( lsvc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_df = pd.concat( cv_dict, axis = 1).round( 4)
    return cv_df

def print_best_parameters( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict):
    best_params_dict = {}
    best_params_dict[ 'KNN'] = knn_dict[ 'best_params']
    best_params_dict[ 'SVC'] = svc_dict[ 'best_params']
    best_params_dict[ 'RFC'] = rfc_dict[ 'best_params']
    best_params_dict[ 'NB'] = nb_dict[ 'best_params']
    best_params_dict[ 'LogReg'] = logreg_dict[ 'best_params']
    best_params_dict[ 'LinearSVC'] = lsvc_dict[ 'best_params']
    print( pd.DataFrame( best_params_dict).fillna( ''))

def test_scores( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict):
    test_scores_df = pd.DataFrame( {
        'KNN': pd.Series( knn_dict[ 'test_scores']),
        'SVC': pd.Series( svc_dict[ 'test_scores']),
        'RFC': pd.Series( rfc_dict[ 'test_scores']),
        'NB': pd.Series( nb_dict[ 'test_scores']),
        'LogReg': pd.Series( logreg_dict[ 'test_scores']),
        'LinearSVC': pd.Series( lsvc_dict[ 'test_scores'])
    }).round( 4)
    return test_scores_df

if __name__ == '__main__':
    main()