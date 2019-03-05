import numpy as np

def defaultUPrior(col_len):
	uprior = np.empty((2,col_len))
	uprior[0] = -np.inf
	uprior[1] =  np.inf
	#
	return uprior

def defaultGPrior(col_len):
	gprior = np.empty((2,col_len))
	gprior[0] = 0.0
	gprior[1] = np.inf
	#
	return gprior

def positiveUPrior(col_len):
	uprior = np.empty((2,col_len))
	uprior[0] = 0.0
	uprior[1] = np.inf
	#
	return uprior

def negativeUPrior(col_len):
	uprior = np.empty((2,col_len))
	uprior[0] = -np.inf
	uprior[1] = 0.0
	#
	return uprior

def checkUPrior(v, vname, vsize=None):
	if v is None: return None
	assert v.shape[0]==2, vname+': num of rows must be 2.'
	assert np.all(v[0]<=v[1]),\
		vname+': ubounds must be greater or equal than lbounds.'
	if vsize is not None:
		assert v.shape[1]==vsize, vname+': wrong num of cols.'

def checkGPrior(v, vname, vsize=None):
	if v is None: return None
	assert v.shape[0]==2, vname+': num of rows must be 2.'
	assert np.all(v[1]>0.0), vname+': variance must be positive.'
	if vsize is not None:
		assert v.shape[1]==vsize, vname+': wrong num of cols.'