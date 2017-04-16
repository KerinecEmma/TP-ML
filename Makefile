MAINFILE=main

all: $(MAINFILE).py
	python3 $^ circle Adaboost -n 100



clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc  __pycache*