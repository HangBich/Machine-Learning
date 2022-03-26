#!pip install pytorch-tabnet wget

from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)
from pytorch_tabnet.tab_model import TabNetClassifier


import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt
%matplotlib inline


def predict(pred):
    return [1 if i>0.5 else 0 for i in pred]
    
if __name__ == "__main__":
	df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
	df.head()
	X = df.iloc[:, 3:13]
	y = df.iloc[:,13]
	labelEncoder = LabelEncoder()
	X['Geography'] = labelEncoder.fit_transform(X['Geography'])
	X['Gender'] = labelEncoder.fit_transform(X['Gender'])

	features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
	       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
	x = torch.tensor(X[features].values, dtype=torch.double).numpy()
	y = torch.tensor(df['Exited'].values, dtype=torch.double).numpy()

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	X_train, X_valid, y_train, y_valid= train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

	classifier = TabNetClassifier(verbose=0,seed=42)
	classifier.fit(X_train=X_train, y_train=y_train,
		       patience=5,max_epochs=100,
		       eval_metric=['auc'])
	test_predictions = classifier.predict_proba(X_test)[:,1]

	test_result = predict(test_predictions)
	print(f'Train accuracy score: {accuracy_score(y_test,test_result)}, Test roc_auc_score {roc_auc_score(y_test,test_result)}')

	train_predictions = classifier.predict_proba(X_train)[:,1]
	train_result = predict(train_predictions)
	print(f'Train accuracy score: {accuracy_score(y_train,train_result)}, Test roc_auc_score {roc_auc_score(y_train,train_result)}')
