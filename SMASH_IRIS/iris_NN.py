from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

data = load_iris(return_X_y=True, as_frame=True)
iris = load_iris()

X,y = data[0], data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

class Model(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,30)
        self.fc2 = nn.Linear(30,out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(lr=0.01, params=model.parameters())
epochs = 400
losses = []

X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.as_tensor(y_train.values, dtype=torch.long).squeeze()

for ep in tqdm(range(epochs)):
    model.train()
    ypred = model(X_train_tensor)
    loss = criterion(ypred, y_train_tensor)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.show()

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)

X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.as_tensor(y_test.values, dtype=torch.long).squeeze()


# test
correct = 0
incorrect = 0

with torch.no_grad():
    model.eval()
    yhat = model(X_test_tensor)
    
    for row in range(len(y_test_tensor)):
        if torch.argmax(yhat[row]) == y_test_tensor[row]:
            correct += 1
        else:
            incorrect += 1

print(f'correct predictions: {correct}\nincorrect:{incorrect}')