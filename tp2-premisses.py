#!/usr/bin/python
# -​*- coding: utf-8 -*​-

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=50,n_features=2, n_redundant=0, n_informative=2,
random_state=2, n_clusters_per_class=1)

S=[]
for i in range(len(X)):
	S.append(((X[i][0],X[i][1]),y[i]))

print(S[0])



"""plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
print(type(X))"""
