#kNN Algorithm for classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris_loc = "~/Documents/datasets/iris/iris.data"
iris_cols = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_df = pd.read_csv(iris_loc, names = iris_cols)

iris_df["class"][iris_df["class"] == "Iris-setosa"] = 1
iris_df["class"][iris_df["class"] == "Iris-versicolor"] = 2
iris_df["class"][iris_df["class"] == "Iris-virginica"] = 3

iris_train, iris_test = train_test_split(iris_df,test_size=0.25)
iris_test, iris_test_sol = iris_test[["sepal-length", "sepal-width", "petal-length", "petal-width"]], iris_test["class"]

X_train = iris_train[["sepal-length", "sepal-width", "petal-length", "petal-width"]].values
Y_train = iris_train["class"].values

#How many neighbors do we want to analyze for kNN
k = 5

#We Want to normalize our feature vectors so that inner products correspond to projection onto feature space

for i in range(len(X_train)):
    X_train[i] = X_train[i]/np.linalg.norm(X_train[i])

X_test = iris_test.values

#Project each vector we want to classify onto the feature space
test_projection = X_test.dot(X_train.T)

Y_test = []

for i in range(len(X_test)):
    sorted_args = np.argsort(test_projection[i])
    neighbors_val = [Y_train[sorted_args[-i]] for i in range(1,k+1)]
    Y_test.append(np.argmax(np.bincount(neighbors_val)))

total_misclassified = 0

for i in range(len(X_test)):
    if Y_test[i] - iris_test_sol.values[i] != 0:
        total_misclassified += 1

error_percent = total_misclassified / len(X_test) * 100

print("Error:" + str(error_percent) + "%")
