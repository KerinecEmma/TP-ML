from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


x=(10,0)
"""int donnee a classer"""

"""donnees et leurs classes"""
X, y = make_classification(n_samples=50,n_features=2, n_redundant=0, n_informative=2,
random_state=2, n_clusters_per_class=1)

a=X[len(X)-1]
b=y[len(X)-1]

S=[]
for i in range(len(X)-1):
	S.append(((X[i][0],X[i][1]),y[i]))


def d(x,y):	
	"""fonction de distance, prend deux donnees"""
	return((x[0]-y[0])**2 + (x[1]-y[1])**2)

k=45
"""nombre de voisins utilise"""

neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X, y) 
print(neigh.predict([a]))

def knn(x, S, d, k):
	"""version 1D, marche"""
	tab=[(1,1)]*len(S)
	"""tableau de distance à x, et classe"""
	for i in range(len(S)):
		tab[i]=(d(x, (S[i])[0]),(S[i])[1])
	"""trie(tab suivant prem coord increasing order)"""
	tab.sort(key=lambda tup: tup[0])
	classes=[0,0]
	for i in range(k):
		if tab[i][1]==0:
			classes[0]+=1
		else:
			classes[1]+=1
	if classes[0]>=classes[1]:
		return(0)
	else:
		return(1)
	
print(knn(a, S, d, k))	






	

