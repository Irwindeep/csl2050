import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Reading the test and train files to load the transformed dataset in form of DataFrames
df = pd.read_csv("./train.csv", header=None)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

df = pd.read_csv("./test.csv", header=None)
X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]

# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Accuracy for Naive Bayes Classifier: {np.mean(gnb.predict(X_test) == y_test)*100:.2f}%")

# K-Nearest neighbors classifier with k = 5
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
print(f"Accuracy for K-Nearest Neighbor Classifier: {np.mean(knn.predict(X_test) == y_test)*100:.2f}%")

# Random Forets Classifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
print(f"Accuracy for Random Forest Classifier: {np.mean(forest.predict(X_test) == y_test)*100:.2f}%")

# Support Vector Classifier with linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print(f"Accuracy for Support Vector Classifier with linear kernel = {np.mean(svm.predict(X_test) == y_test)*100:.2f}%")

# Converting the train and test data to tensors
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))

# ANN class
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # defining the sequence of layers
        self.layers = nn.Sequential(
            nn.Linear(X_train.shape[1], 200),
            nn.ReLU(),  # ReLU activation in each hidden layer
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, y_train.unique().shape[0])
        )
    # Forwarding the input to the layers
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
# Defining train dataset and train loader for batch size of 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Calling the model
model = ANN()
criterion = nn.CrossEntropyLoss()   # Cross Entropy Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer

# training the model for 20 epochs
for epoch in range(20):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{20}], Loss: {loss.item()}")

with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)  # Predictions for test data
    probabilities = nn.functional.softmax(outputs, dim=1)   # Applying softmax activation for output layer
    _, predicted = torch.max(probabilities, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)   # Computing accuracy of the model
    print(f"Accuracy for Neural Network with Cross Entropy Loss as Loss function = {accuracy*100:.2f}%")