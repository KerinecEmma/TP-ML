#!/usr/bin/python
# -​*- coding: utf-8 -*​-

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np



"""pas du tout fini, faire les structures, comprendre m, Z..."""

def adaboost(S, t, L):
	m=len(S)
	for i in range(1, m):
		d[1][i]=1/m
	for i in range(1, t):
		h[i]=L(S,D[i]
		e[i]=
		"""fonction a implementer"""
		a[i]=1/2 np.log((1-e[i])/e[i])
		for j in range(1, m):
			d[i+1][j]=d[i][j]*exp(a[j]*y[i]*h[j][i]))/z[i]
			"""z[i] is a normalization coefficient"""
	f(x)=
	
