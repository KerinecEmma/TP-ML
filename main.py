#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import generate_tests as gtest
import argparse
import ml_algorithms as alg

def main():
	parser = argparse.ArgumentParser(description='Machine Learning - TP2')
	parser.add_argument("dataset", type=str, choices=["classi","gauss","moon","circle","iris","digits"])
	parser.add_argument("-n", metavar="N", type=int)
	parser.add_argument("algorithm", type=str, choices=["kNN"])
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

	elif args.dataset=="iris":
		X, y=gtest.generate_iris()

	elif args.dataset=="digits":
		X, y=gtest,generate_digits()

	if args.algorithm=="kNN":
		data_classified=alg.kNN(X, y)

		plt.plot()


if __name__=='__main__':
	main()