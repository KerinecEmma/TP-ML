from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier



class Adaboost :
	def __init__(self, X_train, y_train, T):
		self._T=T
		for i in range(len(y_train)):
			if y_train[i]==0:
				y_train[i]=-1
		m=len(y_train)
		d=[1/m]*m
		h=[]
		for i in range(T):
			H = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
			h+=[H]
		a=[0]*m

		for t in range(T):
			e=0
			h[t].fit(X_train, y_train.reshape(-1,1), sample_weight=d)

			for i in range(m):
				if (y_train[i] != h[t].predict(X_train[i].reshape(1,-1))):
					e+=d[i]

			if e>0.5 or e == 0:
				self._T=t
				break
			else:
				a[t]=math.log((1-e)/e)/2

			z=0
			for i in range(m):
				d[i]=d[i]*math.exp(-a[t]*y_train[i]*(h[t].predict(X_train[i].reshape(1,-1))))
				z+=d[i]

			for i in range(m):
				d[i]= d[i]/z
		self._a=a
		self._h=h

	def predict(self, x):
		f=0
		for t in range(self._T):
			f+= self._a[t] * self._h[t].predict(x.reshape(1,-1))

		return (np.sign(f))
	
