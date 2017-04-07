MAINFILE=main

all: $(MAINFILE).py
	python3 $^ moon kNN -n 50



clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc  __pycache*