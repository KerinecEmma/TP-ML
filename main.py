#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import generate_tests as gtest
import argparse
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import math
from tp2_knn import *
from tp2_boosting import *



def show_scores(X, X_test, y_test, X_train, y_train, Y_learn_Pyth, Y_learn_Keri, pyth_learn, ker_learn):
	h = .02  # grid step
	x_min= X[:, 0].min() - 1
	x_max= X[:, 0].max() + 1
	y_min = X[:, 1].min() - 1
	y_max = X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



	X_graph=X_test[:,0]
	Y_graph=X_test[:,1]
	f,((sub1, sub2), (sub3, sub4))=plt.subplots(2,2)

	sub1.plot()
	Z2d = pyth_learn.predict(np.c_[xx.ravel(),yy.ravel()])
	Z2d=Z2d.reshape(xx.shape)
	sub1.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
	sub1.scatter(X_graph, Y_graph, c=Y_learn_Pyth, cmap=plt.cm.coolwarm)
	sub1.set_title("Python's Learnt")
	sub1.axis([x_min,x_max,y_min,y_max])


	if ker_learn!=None :
		sub2.plot()
		grid=np.c_[xx.ravel(),yy.ravel()]
		Z2d=[]
		for i in range(len(grid)):
			Z2d+=[ker_learn.predict(grid[i])]
		Z2d=np.array(Z2d).reshape(xx.shape)
		sub2.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
		sub2.scatter(X_graph, Y_graph, c=Y_learn_Keri, cmap=plt.cm.coolwarm)
		sub2.set_title("Our implementation")
		sub2.axis([x_min,x_max,y_min,y_max])


	sub3.plot()
	sub3.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap=plt.cm.coolwarm)
	sub3.set_title("Training Sample")
	sub3.axis([x_min,x_max,y_min,y_max])


	sub4.plot()
	sub4.scatter(X_test[:,0], X_test[:,1],c=y_test, cmap=plt.cm.coolwarm)
	sub4.set_title("Test Sample")
	sub4.axis([x_min,x_max,y_min,y_max])


	plt.show(block=False)



def main():
	parser = argparse.ArgumentParser(description='Machine Learning - TP2')
	parser.add_argument("dataset", type=str, choices=["classi","gauss","moon","circle","iris","digits","ozone"])
	parser.add_argument("-n", metavar="N", type=int)
	parser.add_argument("-print", action='store_const', const=1,)
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

	elif args.dataset=="ozone":
		X, y=gtest.generate_ozone()


	kf=sklearn.model_selection.KFold(n_splits=5,shuffle=True) #creation of the k-folds
	scoresKeri=[]
	scoresPyth=[]
	tot=[]

	## Running algorithms
	if args.algorithm=="kNN":		
		for k in range(1,math.floor(0.75*num_ex)):
			for learn,test in kf.split(X):
				X_train=X[learn]
				y_train=y[learn]
				X_test=X[test]
				y_test=y[test]
				Y_learn_Keri=y_test.copy()
				Y_learn_Pyth=y_test.copy()
				neigh = KNeighborsClassifier(n_neighbors=k)
				train=[(X_train[i], y_train[i]) for i in range(len(X_train))]
				ker_neig = kNN(k, train, d)
				neigh.fit(X_train, np.ravel(y_train.reshape(-1,1)))
				score=0
				score1=0
				tpt=0
				for i in range(len(X_test)):
					Y_learn_Pyth[i]=neigh.predict(X_test[i].reshape(1,-1))
					Y_learn_Keri[i]=ker_neig.predict(X_test[i])
					tpt+=1
					score+=Y_learn_Pyth[i]==y_test[i]
					score1+=Y_learn_Keri[i]==y_test[i]
			scoresKeri.append(score1)
			scoresPyth.append(score)
			tot.append(tpt)
			if args.print:
				show_scores(X, X_test, y_test, X_train, y_train, Y_learn_Pyth, Y_learn_Keri, neigh, ker_neig)

		#Outputs
		for i in range(len(scoresKeri)):
			print("kNN with k = {}, Emma has {} right vs Python {}".format(5*i+1,scoresKeri[i]/tot[i],scoresPyth[i]/tot[i]))

	elif args.algorithm=="Adaboost":
		for k in range(1,math.floor(0.75*num_ex)):
			for learn,test in kf.split(X):
				X_train=X[learn]
				y_train=y[learn]
				X_test=X[test]
				y_test=y[test]
				Y_learn_Keri=y_test.copy()
				Y_learn_Pyth=y_test.copy()
				dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
				ada = sklearn.ensemble.AdaBoostClassifier(base_estimator=dt_stump, n_estimators=k)
				ada.fit(X_train, np.ravel(y_train.reshape(-1,1)))
				adaker=Adaboost(X_train, y_train, k)
				score=0
				score1=0
				tpt=0
				for i in range(len(X_test)):
					Y_learn_Pyth[i]=ada.predict(X_test[i].reshape(1,-1))
					Y_learn_Keri[i]=adaker.predict(X_test[i])
					score+=Y_learn_Pyth[i]==y_test[i]
					score1+=Y_learn_Keri[i]==y_test[i]
					tpt+=1
			scoresKeri.append(score1)
			scoresPyth.append(score)
			tot.append(tpt)
			if args.print:
				show_scores(X, X_test, y_test, X_train, y_train, Y_learn_Pyth, Y_learn_Keri, ada, adaker)

			

		#Outputs
		for i in range(len(scoresKeri)):
			print("Adaboost with n = {}, Emma has {} right vs Python {}".format(5*i+1,scoresKeri[i]/tot[i],scoresPyth[i]/tot[i]))

	elif args.algorithm=="SVM":
		kernel=[ "linear", "poly", "rbf", "sigmoid"]
		for c in [0.4, 0.6, 0.8, 1, 1.5 , 2]:
			scoresPyth=[]
			tot=[]
			for k in kernel:
				for learn,test in kf.split(X):
					X_train=X[learn]
					y_train=y[learn]
					X_test=X[test]
					y_test=y[test]
					Y_learn_Pyth=y_test.copy()
					svm = sklearn.svm.SVC(kernel=k, C=c)
					svm.fit(X_train, np.ravel(y_train.reshape(-1,1)))
					score=0
					tpt=0
					for i in range(len(X_test)):
						Y_learn_Pyth[i]=svm.predict(X_test[i].reshape(1,-1))
						score+=Y_learn_Pyth[i]==y_test[i]
						tpt+=1
				scoresPyth.append(score)
				tot.append(tpt)
				if args.print:
					show_scores(X, X_test, y_test, X_train, y_train, Y_learn_Pyth, None, svm, None)

			for i in range(len(scoresPyth)):
				print("SVM with kernel = {} and c ={}, Python has {} number of right entries".format(kernel[i], c,scoresPyth[i]/tot[i]))

	input("Press Enter to continue...")



if __name__=='__main__':
	main()