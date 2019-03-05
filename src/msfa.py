# src file of the pysfa class

import numpy as np


from numpy             import exp, log, sqrt, pi
from numpy.linalg      import det, norm, solve
from scipy.special     import erf, erfc
from scipy.optimize    import bisect
from sfa_utils.npufunc import log_erfc, special
from pybs import *

from msfa_prior import *
from msfa_utils import special


# SFA Main Class
# =============================================================================
class MSFA:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, X, Z, D, S, Y=None, vtype='half_normal', ftype='upper',
			add_intercept_to_x=False,
			add_intercept_to_z=False,
			add_intercept_to_d=False):
		# add intercept
		if add_intercept_to_x: X = np.insert(X, 0, 1.0, axis=1)
		if add_intercept_to_z: Z = np.insert(Z, 0, 1.0, axis=1)
		if add_intercept_to_d: D = np.insert(D, 0, 1.0, axis=1)
		# pass in the data
		self.X = X
		self.Y = Y
		self.Z = Z
		self.D = D
		self.S = S
		self.V = S**2
		# pass in the dimension
		self.updateDim()
		#
		self.vtype = vtype
		self.ftype = ftype
		
		#
		self.check()
		#
		self.defaultOptions()

	# update functions
	# -------------------------------------------------------------------------
	def updateDim(self):
		self.N = self.X.shape[0]
		self.k_beta = self.X.shape[1]
		self.k_gama = self.Z.shape[1]
		self.k_deta = self.D.shape[1]
		self.k = self.k_beta + self.k_gama + self.k_deta
		#
		self.id_beta = slice(self.k_beta)
		self.id_gama = slice(self.k_beta, self.k_beta + self.k_gama)
		self.id_deta = slice(self.k_beta + self.k_gama, self.k)

	# check the data consistancy
	# -------------------------------------------------------------------------
	def check(self):
		# check input data dimension
		assert len(self.X)==self.N, 'X: wrong number of rows.'
		assert len(self.Z)==self.N, 'Z: wrong number of rows.'
		assert len(self.D)==self.N, 'D: wrong number of rows.'
		assert len(self.S)==self.N, 'S: wrong number of elements.'
		#
		if self.Y is not None:
			assert len(self.Y)==self.N, 'Y: wrong number of measurements.'
		#
		# check the vtype
		assert self.vtype in ['half_normal', 'exponential'], \
			'vtype: only support half_normal and exponential vtype.'
		# check the ftype
		assert self.ftype in ['upper', 'lower'], \
			'ftype: only support upper and lower ftype.'
		#
		# check if all elements in Z is non-negative
		assert np.all(self.Z>=0.0), 'Z: all elements must be non-negative.'
		assert np.all(self.D>=0.0), 'D: all elements must be non-negative.'
		#
		# check if the system is identifiable
		ux, sx, vx = np.linalg.svd(self.X)
		uz, sz, vz = np.linalg.svd(self.Z)
		ud, sd, vd = np.linalg.svd(self.D)
		#
		assert sx[-1] > 1e-10, 'X: model mis-specification, is singular.'
		assert sz[-1] > 1e-10, 'Z: model mis-specification, is singular.'
		assert sd[-1] > 1e-10, 'D: model mis-specification, is singular.'
		#
		cx = sx[0]/sx[-1]
		cz = sz[0]/sz[-1]
		cd = sd[0]/sd[-1]
		#
		# print out the data summary
		print('number of measurements:', self.N)
		print('dimension of beta:     ', self.k_beta)
		print('dimension of gama:     ', self.k_gama)
		print('dimension of deta:     ', self.k_deta)
		print('cond number of X cov:  ', cx)
		print('cond number of Z cov:  ', cz)
		print('cond number of D cov:  ', cd)

	# default options
	# -------------------------------------------------------------------------
	def defaultOptions(self):
		# default not using trimming
		self.use_trimming = False
		#
		# default not using bspline
		self.bspline_uprior = None
		self.bspline_gprior = None
		self.bspline_degree = None
		self.bspline_knots  = None
		self.bspline_dpmat  = None
		self.bspline_d1mat  = None
		self.bspline_d2mat  = None
		#
		# default not having constrains
		self.use_constraint = False
		self.num_constraint = 0
		self.constraint_matrix = np.array([]).reshape(0, self.k)
		self.constraint_values = np.array([]).reshape(2, 0)
		#
		# default not have quadratic penalties
		self.use_qpenalty = False
		self.qpenalty_matrix = np.array([]).reshape(0, self.k)
		self.qpenalty_values = np.array([]).reshape(2, 0)
		#
		# default solution
		self.soln = None
		self.info = 'no solution, haven\'t run the solver yet.'
		self.beta_soln = None
		self.gama_soln = None
		self.deta_soln = None
		#
		# add default uprior (constrains)
		self.use_uprior = True
		self.beta_uprior = defaultUPrior(self.k_beta)
		self.gama_uprior = positiveUPrior(self.k_gama)
		self.deta_uprior = positiveUPrior(self.k_deta)
		self.updateUPrior()
		#
		self.use_gprior = False
		self.beta_gprior = defaultGPrior(self.k_beta)
		self.gama_gprior = defaultGPrior(self.k_gama)
		self.deta_gprior = defaultGPrior(self.k_deta)
		self.updateGPrior()

	# add uniform prior
	# -------------------------------------------------------------------------
	def addUPrior(self, beta_uprior, gama_uprior, deta_uprior):
		# check input
		checkUPrior(beta_uprior, 'beta_uprior', vsize=self.k_beta)
		checkUPrior(gama_uprior, 'gama_uprior', vsize=self.k_gama)
		checkUPrior(deta_uprior, 'deta_uprior', vsize=self.k_deta)
		#
		assert np.all(gama_uprior[0]>=0.0), \
			'gama_uprior: lbounds for variance must be non-negative'
		assert np.all(deta_uprior[0]>=0.0), \
			'deta_uprior: lbounds for variance must be non-negative'
		#
		if beta_uprior is not None: self.beta_uprior = beta_uprior
		if gama_uprior is not None: self.gama_uprior = gama_uprior
		if deta_uprior is not None: self.deta_uprior = deta_uprior
		#
		self.updateUPrior()
		#
		self.use_uprior = True

	def updateUPrior(self):
		self.uprior = np.hstack((
			self.beta_uprior,
			self.gama_uprior,
			self.deta_uprior
			))

	# add gaussian prior
	# -------------------------------------------------------------------------
	def addGPrior(self, beta_gprior, gama_gprior, deta_uprior):
		# check input
		checkGPrior(beta_gprior, 'beta_gprior', vsize=self.k_beta)
		checkGPrior(gama_gprior, 'gama_gprior', vsize=self.k_gama)
		checkGPrior(deta_gprior, 'deta_gprior', vsize=self.k_deta)
		#
		if beta_gprior is not None: self.beta_gprior = beta_gprior
		if gama_gprior is not None: self.gama_gprior = gama_gprior
		if deta_gprior is not None: self.deta_gprior = deta_gprior
		#
		self.updateGPrior()
		#
		self.use_gprior = True

	def updateGPrior(self):
		self.gprior = np.hstack((
			self.beta_gprior,
			self.gama_gprior,
			self.deta_gprior
			))

	# add linear constraint
	# -------------------------------------------------------------------------
	def addConstraint(self, C, c, col_id):
		A = C
		C = np.zeros((A.shape[0], self.k))
		C[:,col_id] = A
		#
		self.constraint_matrix = np.vstack((self.constraint_matrix, C))
		self.constraint_values = np.hstack((self.constraint_values, c))
		self.num_constraint = self.constraint_matrix.shape[0]
		#
		self.use_constraint = True

	# add quadratic penalites
	# -------------------------------------------------------------------------
	def addQuadPenalty(self, C, c, w, col_id):
		A = C
		C = np.zeros((A.shape[0], self.k))
		C[:,col_id] = A
		#
		self.qpenalty_matrix = np.vstack((self.qpenalty_matrix, C))
		self.qpenalty_values = np.hstack((self.qpenalty_values, c))
		#
		self.use_qpenalty = True
	
	# add bspline uniform prior
	# -------------------------------------------------------------------------
	def addBSplineUPrior(self, bspline_uprior):
		k = len(self.bspline_knots) - 1
		# check input
		checkUPrior(bspline_uprior, 'bspline_uprior', vsize=k)
		#
		self.bspline_uprior = bspline_uprior
		# create constraints
		self.addConstraint(self.bspline_dpmat, self.bspline_uprior,
			self.id_beta_bs)

	# add bspline gaussian prior
	# -------------------------------------------------------------------------
	def addBSplineGPrior(self, bspline_gprior):
		k = len(self.bspline_knots) - 1
		# check input
		checkUPrior(bspline_uprior, 'bspline_uprior', vsize=k)
		#
		self.bspline_gprior = bspline_gprior
		# create qpenalty
		self.addQuadPenalty(self.bspline_dpmat, self.bspline_gprior,
			self.id_beta_bs)

	# add trimming
	# -------------------------------------------------------------------------
	def addTrimming(self, h):
		# check if h is valid
		assert 0<h<self.N, 'h should be positive and strictly less than N.'
		#
		self.h = h
		self.w = np.repeat(self.h/self.N, self.N)
		#
		print('trim %0.3f of the data' % (1.0 - self.h/self.N))
		#
		self.use_trimming = True

	# delete x covariates
	# -------------------------------------------------------------------------
	def delXCov(self, col_id):
		# delete X cov
		self.X = np.delete(self.X, col_id, 1)
		# delete u prior
		self.beta_uprior = np.delete(self.beta_uprior, col_id, 1)
		self.updateUPrior()
		# delete g prior
		self.beta_gprior = np.delete(self.beta_gprior, col_id, 1)
		self.updateGPrior()
		# reset dimension
		self.updateDim()
		# delete qpenalty
		if self.qpenalty_matrix.size == 0:
			self.qpenalty_matrix = np.array([]).reshape(0, self.k)
		else:
			self.qpenalty_matrix = np.delete(self.qpenalty_matrix, col_id, 1)
		# delete constraint
		if self.constraint_matrix.size == 0:
			self.constraint_matrix = np.array([]).reshape(0, self.k)
		else:
			self.constraint_matrix = np.delete(self.constraint_matrix,
				col_id, 1)

	# add x covariates
	# -------------------------------------------------------------------------
	def addXCov(self, X, col_id):
		# add X cov
		self.X = np.insert(self.X, col_id, X, 1)
		# delete u prior
		beta_uprior = defaultUPrior(X.shape[1])
		self.beta_uprior = np.insert(self.beta_uprior, col_id, beta_uprior, 1)
		self.updateUPrior()
		# delete g prior
		beta_gprior = defaultGPrior(X.shape[1])
		self.beta_gprior = np.insert(self.beta_gprior, col_id, beta_gprior, 1)
		self.updateGPrior()
		# reset dimension
		self.updateDim()
		# add qpenalty
		if self.qpenalty_matrix.size == 0:
			self.qpenalty_matrix = np.array([]).reshape(0, self.k)
		else:
			self.qpenalty_matrix = np.insert(self.qpenalty_matrix,
				col_id, 0.0, 1)
		# add constraint
		if self.constraint_matrix.size == 0:
			self.constraint_matrix = np.array([]).reshape(0, self.k)
		else:
			self.constraint_matrix = np.insert(self.constraint_matrix,
				col_id, 0.0, 1)

	# add bspline
	# -------------------------------------------------------------------------
	def addBSpline(self, knots, degree, col_id=1,
			l_linear=False,
			r_linear=False,
			bspline_uprior=None,
			bspline_gprior=None,
			bspline_mono=None,
			bspline_cvcv=None):
		# check if there enough columns in x cov
		assert self.k_beta>=2, 'no x cov for bspline.'
		#
		self.bspline_knots  = knots
		self.bspline_degree = degree
		#
		# create design matrix
		X, bs = designMat(self.X[:,col_id], knots, degree,
					l_linear=l_linear, r_linear=r_linear)
		#
		self.k_beta_bs = X.shape[1]
		self.id_beta_bs = slice(self.k_beta_bs)
		#
		self.updateBSDmat(bs, l_linear, r_linear)
		#
		# replace the X cov
		del_id = [col_id]
		if np.all(self.X[:,0]==1.0): del_id.append(0)
		self.delXCov(del_id)
		self.addXCov(X, [0])
		#
		# add priors
		if bspline_uprior is not None: self.addBSplineUPrior(bspline_uprior)
		if bspline_gprior is not None: self.addBSplineGPrior(bspline_gprior)
		#
		self.updateBSShape(bspline_mono, bspline_cvcv)

	def updateBSDmat(self, bs, l_linear, r_linear):
		degree = self.bspline_degree
		self.bspline_dpmat = bs.derivativeMat(degree, degree)
		self.bspline_d1mat = bs.derivativeMat(1, degree)
		self.bspline_d2mat = bs.derivativeMat(2, degree)
		if l_linear:
			l_dpmat = np.zeros(self.k_beta_bs)
			l_dpmat[0] = 1.0/(bs.t[0] - bs.t[1])
			l_dpmat[1] = 1.0/(bs.t[1] - bs.t[0])
			self.bspline_dpmat = np.vstack((l_dpmat, self.bspline_dpmat))
		if r_linear:
			r_dpmat = np.zeros(self.k_beta_bs)
			r_dpmat[-2] = 1.0/(bs.t[-2] - bs.t[-1])
			r_dpmat[-1] = 1.0/(bs.t[-1] - bs.t[-2])
			self.bspline_dpmat = np.vstack((self.bspline_dpmat, r_dpmat))

	def updateBSShape(self, bspline_mono, bspline_cvcv):
		# add constraints
		if bspline_mono == 'increasing':
			self.addConstraint(self.bspline_d1mat,
				positiveUPrior(self.k_beta_bs - 1), self.id_beta_bs)
		if bspline_mono == 'decreasing':
			self.addConstraint(self.bspline_d1mat,
				negativeUPrior(self.k_beta_bs - 1), self.id_beta_bs)
		#
		if bspline_cvcv == 'convex':
			self.addConstraint(self.bspline_d2mat,
				positiveUPrior(self.k_beta_bs - 2), self.id_beta_bs)
		if bspline_cvcv == 'concave':
			self.addConstraint(self.bspline_d2mat,
				negativeUPrior(self.k_beta_bs - 2), self.id_beta_bs)

	# data simulation
	# -------------------------------------------------------------------------
	def simData(self, beta_t, gama_t, deta_t):
		# check input dimension
		assert len(beta_t)==self.k_beta, 'beta_t: inconsistant with k_beta'
		assert len(gama_t)==self.k_gama, 'gama_t: inconsistant with k_gama'
		assert len(deta_t)==self.k_deta, 'deta_t: inconsistant with k_deta'
		#
		self.E = np.random.randn(self.N)*self.S
		self.EU = np.random.randn(self.N)*sqrt(self.Z.dot(gama_t))
		if self.vtype == 'half_normal':
			self.EV = np.abs(np.random.randn(self.N)*\
				sqrt(self.D.dot(deta_t)))
		if self.vtype == 'exponential':
			self.EV = np.random.exponential(size=self.N)*\
				sqrt(self.D.dot(deta_t))
		self.Y = self.X.dot(beta_t) + self.EU - self.EV + self.E

	# likelihood function
	# -------------------------------------------------------------------------
	def likelihood(self, x):
		beta = x[self.id_beta]
		gama = x[self.id_gama]
		deta = x[self.id_deta]
		# residual
		R = self.Y - self.X.dot(beta)
		# variance of u and v
		Vu = self.Z.dot(gama)
		Vv = self.D.dot(deta)
		Su = np.sqrt(Vu)
		Sv = np.sqrt(Vv)
		#
		V1 = self.V + Vu
		V2 = self.V + Vu + Vv
		#
		if self.ftype == 'upper':
			L = 0.5*R**2/V2 + 0.5*np.log(2.0*np.pi*V2) - \
				self.log1mErf(R*Sv/np.sqrt(2.0*V1*V2))
		if self.ftype == 'lower':
			L = 0.5*R**2/V2 + 0.5*np.log(2.0*np.pi*V2) - \
				self.log1pErf(R*Sv/np.sqrt(2.0*V1*V2))
		#
		return L

	# objective function
	# -------------------------------------------------------------------------
	def objective(self, x):
		L = self.likelihood(x)
		f = np.sum(L)
		#
		if self.use_gprior:
			f += (x - self.gprior[0])**2/self.gprior[1]
		#
		if self.use_qpenalty:
			f += (self.qpenalty_matrix.dot(x) - self.qpenalty_values[0])/ \
				self.qpenalty_values[1]
		#
		return f

	# ultility functions
	# -------------------------------------------------------------------------
	def log1pErf(self, x):
		if np.isrealobj(x):
			return special.rlog_1p_erf(x.size, x)
		if np.iscomplexobj(x):
			return special.clog_1p_erf(x.size, x)

	def log1mErf(self, x):
		if np.isrealobj(x):
			return special.rlog_1m_erf(x.size, x)
		if np.iscomplexobj(x):
			return special.clog_1m_erf(x.size, x)

