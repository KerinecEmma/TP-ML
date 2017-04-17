import matplotlib.pyplot as plt
import generate_tests
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import KFold
import numpy as np
import csv

def generate_classi(N):
	#random_state is optional, it is a seed
	#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	return make_classification(n_samples=N, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

def generate_gauss(N):
	return make_gaussian_quantiles(n_samples = N, n_features=2, n_classes=3)

def generate_moon(N):
	#plt.scatter(X[:,0],X[:,1], c = y, s = 100)
	return make_moons(n_samples=N, noise = 0.1)

def generate_circle(N):
	#plt.scatter(X[:,0],X[:,1], c = y, s = 100)
	return make_circles(n_samples=N, factor=.3, noise=.05)

def generate_iris():
	return load_iris(return_X_y=True)

def generate_digits():
	return load_digits(return_X_y=True)

def generate_ozone():

	data = csv.reader(open('ozone.dat'))
	X=np.array([])
	y=[]
	# Aix = 1 0 0 0 0
	# Als = 0 1 0 0 0
	# Cad = 0 0 1 0 0
	# Ram = 0 0 0 1 0
	# Pla = 0 0 0 0 1
	for row in data:
		try :
			X=np.append(X, np.array(([float(row[i]) for i in [0,2,3,4,5,6,7,8,9,10,11,12,13]])))
			y+=[1] if float(row[1])>=150 else [-1]
		except ValueError:
			pass
	X=X.reshape(13,1041)
	X=X.transpose()
	return StandardScaler().fit_transform(X), np.array(y)