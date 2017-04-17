from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def d(x,y):	
	"""fonction de distance, prend deux donnees"""
	return((x[0]-y[0])**2 + (x[1]-y[1])**2)

class kNN:
	def __init__(self, k, S, dist = d):
		self._k=k
		self._S=S.copy()
		self._d=dist

	def predict(self, x):
		"""version 1D, marche"""
		tab=[(1,1)]*len(self._S)
		"""tableau de distance à x, et classe"""
		for i in range(len(self._S)):
			tab[i]=(self._d(x, self._S[i][0]),self._S[i][1])
		"""trie(tab suivant prem coord increasing order)"""
		tab.sort(key=lambda tup: tup[0])
		classes=[0,0]
		for i in range(self._k):
			if tab[i][1]==0:
				classes[0]+=1
			else:
				classes[1]+=1
		if classes[0]>=classes[1]:
			return(0)
		else:
			return(1)
