../ppgfilter.so: setup.py filters.h filters.c utils.h utils.c ppgfilter.c
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf ppgfilter.so build
