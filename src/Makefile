# make file for the fortran module of the mrbert functions

utils: msfa_utils.f95
	python -m numpy.f2py -c msfa_utils.f95 -m msfa_utils

clean:
	rm *.so
	rm -rf *.dSYM