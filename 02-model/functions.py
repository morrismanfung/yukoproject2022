import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
import altair as alt
import seaborn as sns

def better_confusion_matrix( y_test, y_hat, labels = [ 0, 1]):
    df = pd.DataFrame( confusion_matrix( y_test, y_hat, labels = labels))
    df.columns = labels
    df = pd.concat( [ df], axis = 1, keys = ['Predicted'])
    df.index = labels
    df = pd.concat( [df], axis = 0, keys = ['Actual'])
    return df

def pr_curve( model, X_train, X_test, y_train, y_test):
    model.fit( X_train, y_train)
    try:
        proba = model.predict_proba( X_test)[ :, 1]
    except:
        proba = model.decision_function( X_test)
    precision, recall, thresholds = precision_recall_curve( y_test, proba)
    thresholds = np.append( thresholds, 1)
    
    plot_df = pd.DataFrame( {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    })
    
    chart = sns.scatterplot(
        data = plot_df,
        x = 'precision',
        y = 'recall'
    )
    return plot_df, chart

def log_func(x):
    return np.log(x+1)

def test_scoring_metrics( y_test, y_hat, X_test):
    odds = X_test[ 'WinOdds']
    cost = len( y_hat)
    pay = sum( odds * ( y_hat * y_test))
    earning = pay - cost

    metrics = {
        'precision': precision_score( y_test, y_hat),
        'recall': recall_score( y_test, y_hat),
        'f1': f1_score( y_test, y_hat),
        'earning': earning
    }
    return metrics