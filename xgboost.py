#Dataset https://www.kaggle.com/shrutimechlearn/churn-modelling
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
import xgboost as xgb
from xgboost import XGBClassifier
import time
import warnings


dataset = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:,13]

labelEncoder = LabelEncoder()
X['Geography'] = labelEncoder.fit_transform(X['Geography'])
X['Gender'] = labelEncoder.fit_transform(X['Gender'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def objective(space):

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    classifier = xgb.XGBClassifier(n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree']
                            )
    
    classifier.fit(X_train, y_train)
    
    predictions = classifier.predict(X_train)
    acc= accuracy_score(y_train, predictions)
    roc = roc_auc_score(y_train, predictions)
    print('Train accuracy: {}\nTest roc_auc_score: {}'.format((acc*100), (roc*100)))

    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    CrossValMean = accuracies.mean()

    print("CrossValMean:", CrossValMean)

    return{'loss':1-CrossValMean, 'status': STATUS_OK }


space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

print("Best: ", best)

predictions = classifier.predict(X_test)
acc= accuracy_score(y_test, predictions)
roc = roc_auc_score(y_test, predictions)
print('Test accuracy: {}\nTest roc_auc_score: {}'.format((acc*100), (roc*100)))


