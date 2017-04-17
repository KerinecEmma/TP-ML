# TP-ML # TP-ML

usage: main.py [-h] [-n N] [-print]
               {classi,gauss,moon,circle,iris,digits,ozone} {kNN,Adaboost,SVM}

Where N is the size of the dataset, ignored when using ozone or digits.

kNN: 
====
Our version has the same performances than Python's one (tested on circles and moons, kFolds with n_splits=10 k=1 to 0.75*size of dataset).

Adaboost:
=========
Our version has a performance similar to Python's one (same conditions of test, number of weak learners = 1 to 0.75*size of dataset).
It seems a bit more performant: on the circle dataset, it needs less weak learners to completely fit the data.

SVM:
====
Some kernels seem perfectly unadapted to learn the circle dataset, only rbf works. 

Ozone:
======
All methods seem to be good (success rate between 78% and 88%).
The higest success rate is reached with Adaboost with n=71.

Sadly we did not have time to run proper cross validation, but we think that this is quite useless given the high success rate of every method.
