#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import generate_tests as gtest
import argparse
import sklearn
import math
from tp2_knn import *

def main():
	parser = argparse.ArgumentParser(description='Machine Learning - TP2')
	parser.add_argument("dataset", type=str, choices=["classi","gauss","moon","circle","iris","digits"])
	parser.add_argument("-n", metavar="N", type=int)
	parser.add_argument("algorithm", type=str, choices=["kNN", "Adaboost", "SVM"])
	args = parser.parse_args()

	if args.n==None:
		num_ex=50
	else:
		num_ex=args.n

	if args.dataset=="gauss":
		X, y=gtest.generate_gauss(num_ex)

	elif args.dataset=="classi":
		X,y=gtest.generate_classI(num_ex)

	elif args.dataset=="moon":
		X, y=gtest.generate_moon(num_ex)

	elif args.dataset=="circle":
		X, y=gtest.generate_circle(num_ex)

	elif args.dataset=="digits":
		X, y=gtest,generate_digits()

	elif args.dataset=="iris":
		X, y=gtest.generate_iris()

	## Generating examples:
	##X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,y)
	##S=[(X_train[i], y_train[i])  for i in range(X_train)]



	## Running algorithms
	if args.algorithm=="kNN":
		kf=sklearn.model_selection.KFold(n_splits=10,shuffle=True) #creation of the k-folds
		scoresKeri=[]
		scoresPyth=[]
		
		for k in range(1,math.floor(0.75*num_ex),5):
			for learn,test in kf.split(X):
				X_train=X[learn]
				Y_train=y[learn]
				X_test=X[test]
				Y_test=y[test]
				train=[(X_train[i,:], Y_train[i]) for i in range(len(X_train))]
				neigh = KNeighborsClassifier(n_neighbors=k)
				neigh.fit(X_train, Y_train)
				score=0
				score1=0
				for i in range(len(X_test)):
					score+=(kNN(X_test[i], train, d, k)==Y_test[i])
					score1+=(neigh.predict(X_test[i])==Y_test[i])
			scoresKeri.append(score)
			scoresPyth.append(score1)

		#Outputs
		for i in range(len(scoresKeri)):
			print("kNN with k = {}, Emma has {} right vs Python {}".format(5*(i+1),scoresKeri[i],scoresPyth[i]))

	elif args.algorithm=="Adaboost":
		model=ababoost

	elif args.algorithm=="SVM":
		model=SVM




	#TODO plot the data here
	plt.plot()


if __name__=='__main__':
	main()