MAINFILE=main

all: $(MAINFILE).py
	python3 $^ circle kNN -n 100



clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc  __pycache*