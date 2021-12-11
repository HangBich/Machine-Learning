import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score


import numpy as np
import pandas as pd 
import math


#Data 
df = pd.read_csv('/content/sample_data/Churn_Modelling.csv')

#y = df['Exited']
cat_features = [ 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
cont_features = [ 'Balance','EstimatedSalary']
x_cat = torch.tensor(df[cat_features].values, dtype=torch.float32)
x_cont = torch.tensor(df[cont_features].values, dtype=torch.float32)
y = torch.tensor(df['Exited'].values, dtype=torch.float32)

x_catTest = x_cat[:, :]
x_contTest = x_cont[:,:]
y_test = y[:]


#Attention

#initialise weights
hidden_size = (4,4)
dim = 4

w_key=torch.rand(hidden_size, dtype=torch.float32)
w_query=torch.rand(hidden_size, dtype=torch.float32)
w_value=torch.rand(hidden_size, dtype=torch.float32)

query = torch.matmul(x_catTest,w_query)
key = torch.matmul(x_catTest,w_key)
value = torch.matmul(x_catTest,w_value)

attn_scores = torch.matmul(query,key.T)/math.sqrt(dim)
attn_scores_softmax = F.softmax(attn_scores,dim=-1)
weighted_values = value[:,None] * attn_scores_softmax.T[:,:,None]

output = weighted_values.sum(dim=0)
input = torch.cat((output, x_contTest),1)
class MLP(nn.Module):
  def __init__(self,input_size, hidden1_size, hidden2_size, num_classes):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden1_size)
    self.sigmoid1 = nn.Sigmoid()
    self.fc2 = nn.Linear(hidden1_size, hidden2_size)
    self.sigmoid2 = nn.Sigmoid()
    self.fc3 = nn.Linear(hidden2_size, num_classes)
    self.sigmoid3 = nn.Sigmoid()
  def forward(self,x):
    out = self.fc1(x)
    out = self.sigmoid1(out)
    out = self.fc2(out)
    out = self.sigmoid2(out)
    out = self.fc3(out)
    out = self.sigmoid3(out)
    out = out.reshape(-1)
    return out
model = MLP(6, 100, 50, 1)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion  = nn.BCELoss()
x_train, x_test, y_train, y_test = train_test_split(input, y_test, test_size=0.2, random_state=42)
epochs = 20
for epoch in range(epochs):
  optimizer.zero_grad()
  #Forward pass
  y_pred = model(x_train)
  #Compute loss
  # y_pred = y_pred.reshape(-1)
  loss = criterion(y_pred,y_train)
  print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
  #Backward pass
  loss.backward()
  optimizer.step()

model.eval()
y_pred = model(x_test)

ls = []
for i in y_pred:
  if i>0.5:
    ls.append(1)
  else:
    ls.append(0)
accuracy_score(ls,y_test)
#0.8035
roc_auc_score(y_test,ls)
#0.5
