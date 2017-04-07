#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import generate_tests as gtest
import argparse
from tp2_knn import *

def main():
	parser = argparse.ArgumentParser(description='Machine Learning - TP2')
	parser.add_argument("dataset", type=str, choices=["classi","gauss","moon","circle","iris","digits"])
	parser.add_argument("-n", metavar="N", type=int)
	parser.add_argument("algorithm", type=str, choices=["kNN, Adaboost, SVM"])
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

	## Generating examples
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,y)
	S=[(x, y_1)  for x in X_train and y_1 in y_train]



	## Running algorithms
	if args.algorithm=="kNN":
		model=kNN

	elif args.aglorithm=="Adaboost":
		model=ababoost

	elif args.algorithm=="SVM":
		model=SVM

	kf=KFold(n_splits=10,shuffle=True) #creation of the k-folds
	scores=[] 
	for param in range(1,30):
		score=0
		for learn,test in kf.split(X):
			model(X_train, Y_train)
			score = score + model.score(X_test,Y_test)
		scores.append(score)


	#TODO plot the data here
	plt.plot()


if __name__=='__main__':
	main()