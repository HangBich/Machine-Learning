# # Requirements:
# !pip install rtdl
# !pip install libzero==0.0.4

from typing import Any, Dict

import numpy as np
import rtdl
from rtdl import FTTransformer
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing 
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

device = torch.device('cpu')
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)


class PredictModule(nn.Module):
    def __init__(self, shape):
        super(PredictModule, self).__init__()
        self.r, self.c = shape
        self.layers = nn.Sequential(
            nn.LayerNorm([1]),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(self.c, self.r),
            nn.Linear(self.r, 1),
        )
    def forward(self, X):
        return self.layers(X)

def predict(pred):
    return [1 if i>0 else 0 for i in pred]

if __name__=="__main__":

    df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
    X = df.iloc[:, 3:13]
    y = df.iloc[:,13]
    labelEncoder = sklearn.preprocessing.LabelEncoder()
    X['Geography'] = labelEncoder.fit_transform(X['Geography'])
    X['Gender'] = labelEncoder.fit_transform(X['Gender'])
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    cat  = ['Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    # cat  = ['Geography', 'Gender']
    cont = ['CreditScore', 'Balance', 'EstimatedSalary']

    x_cat  = torch.tensor(X[cat].values, dtype=torch.int64)
    x_cont = torch.tensor(X[cont].values, dtype=torch.float32)
    y      = torch.tensor(df['Exited'].values, dtype=torch.float32)

    module = rtdl.FTTransformer.make_baseline(
        n_num_features=3,
        cat_cardinalities=[3,2,74,11,4,2,2],
        d_token=8,
        n_blocks=2,
        attention_dropout=0.2,
        ffn_d_hidden=6,
        ffn_dropout=0.2,
        residual_dropout=0.0,
        d_out=1,
    )

    x = module(x_cont, x_cat)



    model = PredictModule(x.shape)
    # criterion = torch.nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.train()
    epoch = 20


    for epo in range(epoch):
        epoch_loss = []
        optimizer.zero_grad()
        #forward pass
        y_pred = model(X_train)
        test_pred = predict(y_pred)
        
        #Compute loss
        loss = criterion(y_pred.squeeze(), y_train)
        print(f'Train accuracy score: {accuracy_score(y_train,test_pred)}, Train roc_auc_score {roc_auc_score(y_train,test_pred)}')
        #print('Epoch {}: train loss: {}'.format(epo, loss.item()))
        #Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

    model.eval()
    y_pred = model(X_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())

    pred = predict(y_pred)


    print(f'Test accuracy score: {accuracy_score(y_test,pred)}, Test roc_auc_score {roc_auc_score(y_test,pred)}')
