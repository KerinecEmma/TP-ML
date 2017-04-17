MAINFILE=main

all: $(MAINFILE).py
	python3 $^ ozone SVM -n 100



clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc  __pycache*