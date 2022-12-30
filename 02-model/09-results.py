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

from functions import *
from classes import *

def main():
    knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict = read_results()
    cv_df = cv_results( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)
    cv_df.to_csv( '02-model/02-saved-scores/07-cross-validation-results.csv')
    print_best_parameters( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)
    test_scores_df = test_scores( knn_dict, svc_dict, rfc_dict, nb_dict, logreg_dict, lsvc_dict)

    # For voting machine
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']

    pipe_rfc_opt, pipe_logreg_opt, pipe_lsvc_opt = model_training()
    y_hat_rfc_opt = pipe_rfc_opt.predict( X_test)
    y_hat_logreg_opt = pipe_logreg_opt.predict( X_test)
    y_hat_lsvc_opt = pipe_lsvc_opt.predict( X_test)
    vote = y_hat_rfc_opt.astype( 'int') + y_hat_logreg_opt.astype( 'int') + y_hat_lsvc_opt.astype( 'int')
    y_hat_vote = vote > 1
    y_hat_any = vote > 0
    test_scores_vote = test_scoring_metrics( y_test, y_hat_vote, X_test)
    test_scores_any = test_scoring_metrics( y_test, y_hat_any, X_test)
    test_scores_df[ 'Vote'] = pd.Series( test_scores_vote).round( 2)
    test_scores_df[ 'Any'] = pd.Series( test_scores_any).round( 2)

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
    cv_dict[ 'KNN'] = pd.DataFrame( knn_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'SVC'] = pd.DataFrame( svc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'RFC'] = pd.DataFrame( rfc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'NB'] = pd.DataFrame( nb_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'LogReg'] = pd.DataFrame( logreg_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_dict[ 'LinearSVC'] = pd.DataFrame( lsvc_dict[ 'cv_scores']).T.agg( [ 'mean'], axis = 1)
    cv_df = pd.concat( cv_dict, axis = 1).round( 2)
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
    }).round( 2)
    return test_scores_df

# For voting machine
def model_training():
    df_train = pd.read_csv( '01-data/train.csv')
    df_test = pd.read_csv( '01-data/test.csv')
    X_train, y_train = df_train.drop( 'Winner', axis = 1), df_train[ 'Winner']
    X_test, y_test = df_test.drop( 'Winner', axis = 1), df_test[ 'Winner']

    column_transformer = load( '02-model/column_transformer.joblib')
    
    with open( '02-model/02-saved-scores/03-rfc_dict.pkl', 'rb') as f:
        rfc_dict = pickle.load( f)
    best_params_rfc = rfc_dict[ 'best_params']
    thld_rfc = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'RFC_pre'])

    with open( '02-model/02-saved-scores/05-logreg_dict.pkl', 'rb') as f:
        logreg_dict = pickle.load( f)
    best_params_logreg = logreg_dict[ 'best_params']
    thld_logreg = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'LogReg_pre'])

    with open( '02-model/02-saved-scores/06-lsvc_dict.pkl', 'rb') as f:
        lsvc_dict = pickle.load( f)
    best_params_lsvc = lsvc_dict[ 'best_params']
    thld_lsvc = float( pd.read_csv( '02-model/thresholds_used.csv', index_col = 0).loc[ 'LinearSVC_pre'])

    pipe_rfc_opt = final_rfc( column_transformer, best_params_rfc, thld_rfc)
    pipe_logreg_opt = final_logreg( column_transformer, best_params_logreg, thld_logreg)
    pipe_lsvc_opt = final_lsvc( column_transformer, best_params_lsvc, thld_lsvc)

    pipe_rfc_opt.fit( X_train, y_train)
    pipe_logreg_opt.fit( X_train, y_train)
    pipe_lsvc_opt.fit( X_train, y_train)

    return pipe_rfc_opt, pipe_logreg_opt, pipe_lsvc_opt

def final_rfc( column_transformer, best_params, thld):
    pipe_rfc_opt = make_pipeline( column_transformer,
                              RFC_thld( n_estimators = best_params[ 'rfc_thld__n_estimators'],
                                        max_features = best_params[ 'rfc_thld__max_features'],
                                        max_depth = best_params[ 'rfc_thld__max_depth'],
                                        criterion = best_params[ 'rfc_thld__criterion'],
                                        bootstrap = best_params[ 'rfc_thld__bootstrap'],
                                        threshold = thld,
                                        random_state = 918))
    return pipe_rfc_opt

def final_logreg( column_transformer, best_params, thld):
    pipe_logreg_opt = make_pipeline( column_transformer,
                                     LogReg_thld( C = best_params[ 'logreg_thld__C'],
                                                  l1_ratio = best_params[ 'logreg_thld__l1_ratio'],
                                                  threshold = thld,
                                                  random_state = 918))
    return pipe_logreg_opt

def final_lsvc( column_transformer, best_params, thld):
    pipe_lsvc_opt = make_pipeline( column_transformer,
                                   LinearSVC_thld( C = best_params[ 'linearsvc_thld__C'],
                                                   threshold = thld,
                                                   random_state = 918))

    return pipe_lsvc_opt

if __name__ == '__main__':
    main()