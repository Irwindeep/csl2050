import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("train.csv", header=None)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

df = pd.read_csv("test.csv", header=None)
X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for Naive Bayes Classifier: {accuracy*100:.2f}%")

knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for K-Nearest Neighbor Classifier: {accuracy*100:.2f}%")

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for Random Forest Classifier: {accuracy*100:.2f}%")
