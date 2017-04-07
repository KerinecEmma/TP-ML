from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np




#absolument se referer a l'alorithme du cours



def adaboost(x, S, t, L): #learning sample S, number of iterations t, weak learner L. 
	m= len(S)
	d= np.zeros((t, m))
	e= np.zeros(m)
	h= np.zeros(m)

	for i in range(1, m):
		d[1][i]=1/m

	for i in range(1, t):
		h[i]= L(S, d[t])

		for z in range (0, m):
			if (y[z]!= h[i][z]):
				e[i]=+d[i][z]

		a[i]=1/2 np.log((1-e[i])/e[i])

		z=0
		for j in range(1, m):
			d[i+1][j]=d[i][j]*exp(a[j]*y[i]*h[j][i]))
			z=z+d[i+1][j]
		for j in range(1,m): 
			d[i+1][j]=d[i+1][j]/z

	f=0	
	for i in range(1, t):
		f+= a[i] * h[i](x)

	return (np.sign(f))
	
