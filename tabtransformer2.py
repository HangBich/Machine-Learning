import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
  
  
class GEGLU(nn.Module):
    def forward(self,x):
        x, gates = x.chunk(2,dim=-1)
        return x * F.gelu(gates)
        
class FeedForward(nn.Module):
    def __init__(self, dim , mult=5 , dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x, **kwargs):
        return self.net(x)        
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head **-0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q,k,v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
 
class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens = total_tokens,
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x_categ, x_cont):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ += self.categories_offset

        x = self.transformer(x_categ)

        flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim = -1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        x = torch.cat((flat_categ, normed_cont), dim = -1)
        return self.mlp(x)
def predict(pred):
    return [1 if i>0 else 0 for i in pred]
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

labelencoder = LabelEncoder()
df['Gender']=labelencoder.fit_transform(df['Gender'])
df['Geography']=labelencoder.fit_transform(df['Geography'])

features = ['Geography','Gender', 'Tenure', 'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','CreditScore','Balance']
x = torch.tensor(df[features].values, dtype = torch.float32)
y = torch.tensor(df['Exited'].values, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
x_train_cat = X_train[:,0:6]
x_train_cont = X_train[:,6:]

category = []
for i in range(len(x_train_cat[0])):
  category.append(len(x_train_cat[:,i].unique()))
cont_mean_std = torch.randn(3,2)
model = TabTransformer(
    categories = (3,2,11,4,2,2),      # tuple containing the number of unique values within each category
    num_continuous = 3,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                         # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

x_train_cat = x_train_cat.int()
pred_train = model(x_train_cat, x_train_cont)
y_pred = predict(pred_train)
print("Train_accuracy_score: {}\nTrain_roc_auc_score: {}".format( accuracy_score(y_train,y_pred), roc_auc_score(y_train,y_pred)))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train,y_pred)
plt.subplots(1, figsize=(10,10))
plt.title('Train ROC curve')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
    
x_test_cat = X_test[:,0:6]
x_test_cont = X_test[:,6:]

x_test_cat = x_test_cat.int()
pred_test = model(x_test_cat, x_test_cont)
y_pred_test = predict(pred_test)
print("Test_accuracy_score: {}\nTest_roc_auc_score: {}".format( accuracy_score( y_test,y_pred_test), roc_auc_score(y_test,y_pred_test)))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test,y_pred_test)
plt.subplots(1, figsize=(10,10))
plt.title('Test ROC curve')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
    
    
    
