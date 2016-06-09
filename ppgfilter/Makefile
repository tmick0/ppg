../ppgfilter.so: setup.py filters.h filters.c ppgfilter.c
	python setup.py build_ext --inplace
	mv ppgfilter.so ..

.DUMMY: clean

clean:
	rm -rf ../ppgfilter.so build
