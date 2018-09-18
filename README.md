# d9
#scikit-learn

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target   #种类

X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)
knn = KNeighborsClassifier()
knn.fit()
print(knn.predict(X_test)) => [2 1 2 1 0 0 0 0 1 2 0 0 2 1 2 2 0 2 2 2 1 2 2 1 1 0 0 1 0 1 0 2 1 2 2 1 1
                               2 2 2 1 0 1 2 1]
print(y_test) => [2 1 2 1 0 0 0 0 1 2 0 0 2 1 2 2 0 2 2 2 1 2 2 1 1 0 0 1 0 1 0 2 1 2 2 1 1
                  2 2 2 1 0 1 2 2]
                  
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)
print(model.score(data_X,data_y)) => 0.7406077428649428    #百分率
print(model.predict(data_X[:4,:])) => [30.00821269 25.0298606  30.5702317  28.60814055]
print(data_y[:4]) => [24.  21.6 34.7 33.4]

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=1)
plt.scatter(X,y)
plt.show()

