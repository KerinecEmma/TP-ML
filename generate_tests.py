import matplotlib.pyplot as plt
import generate_tests
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import make_gaussian_quantiles


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