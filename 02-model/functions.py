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

'''
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
    
    chart = alt.Chart( plot_df).mark_point().encode(
        x = 'precision',
        y = 'recall',
        tooltip = 'thresholds'
    ).properties( height = 300, width = 300)
    return chart
'''

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
    return chart

def log_func(x):
    return np.log(x+1)

def test_scoring_metrics( y_test, y_hat):
    metrics = {
        'precision': precision_score( y_test, y_hat),
        'recall': recall_score( y_test, y_hat),
        'f1': f1_score( y_test, y_hat)
    }
    return metrics

def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")
# Function by Joel Ostblom