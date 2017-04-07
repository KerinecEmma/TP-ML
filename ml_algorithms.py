import numpy as np

def kNN(k, X, classes, x_training, y_training, distance):
	dst=[]
	for i in range(len(x_training)):
		dst+={distance(x, y, x_training[i], y_training[i]),i}

	dst.sort()

	nb_neighbour=[{0,0},{1,0}]
	for i in range(k):
		nb_neigbour[classes[i]][1]+=1

	nb_neigbour.sort()
	return nb_neigbour[1][0]