# src file of the pysfa class

import numpy as np
import ipopt

from numpy             import exp, log, sqrt, pi
from numpy.linalg      import det, norm, solve
from scipy.special     import erf, erfc
from scipy.optimize    import bisect
from bspline           import *
from sfa_utils.npufunc import log_erfc, special


# SFA Main Class
# =============================================================================
class SFA:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, x, z, d, s, vtype='half_normal', Y=None, n=None,
			add_intercept_to_x=False,
			add_intercept_to_z=False,
			add_intercept_to_d=False):
		# add intercept
		if add_intercept_to_x: x = np.insert(x, 0, 1.0, axis=1)
		if add_intercept_to_z: z = np.insert(z, 0, 1.0, axis=1)
		if add_intercept_to_d: d = np.insert(d, 0, 1.0, axis=1)
		# pass in the parameters
		self.m = x.shape[0]
		self.k_beta = x.shape[1]
		self.k_gama = z.shape[1]
		self.k_deta = d.shape[1]
		self.k = self.k_beta + self.k_gama + self.k_deta
		#
		self.id_beta = slice(0, self.k_beta)
		self.id_gama = slice(self.k_beta, self.k_beta + self.k_gama)
		self.id_deta = slice(self.k_beta + self.k_gama, self.k)
		#
		self.vtype = vtype
		#
		self.x = x
		self.z = z
		self.d = d
		self.s = s
		self.v = s**2
		self.Y = Y
		#
		if n is not None:
			self.n = n
			self.N = np.sum(n)
			self.X = np.repeat(self.x, n, axis=0)
			self.Z = np.repeat(self.z, n, axis=0)
			self.D = np.repeat(self.d, n, axis=0)
			self.S = np.repeat(self.s, n, axis=0)
			self.V = np.repeat(self.v, n, axis=0)
		else:
			self.n = np.ones(self.m)
			self.N = self.m
			self.X = self.x.copy()
			self.Z = self.z.copy()
			self.D = self.d.copy()
			self.S = self.s.copy()
			self.V = self.v.copy()
		#
		self.check()
		#
		self.defaultOptions()

	# check the data consistancy
	# -------------------------------------------------------------------------
	def check(self):
		# check input data dimension
		assert len(self.x)==self.m, 'x: wrong number of rows.'
		assert len(self.z)==self.m, 'z: wrong number of rows.'
		assert len(self.d)==self.m, 'z: wrong number of rows.'
		assert len(self.s)==self.m, 's: wrong number of elements.'
		assert len(self.n)==self.m, 'n: wrong number of elements.'
		#
		if self.Y is not None:
			assert len(self.Y)==self.N, 'Y: wrong number of measurements.'
		#
		# check the vtype
		assert self.vtype in ['half_normal', 'exponential'], \
			'vtype: only support half_normal and exponential vtype.'
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
		print('number of studies:     ', self.m)
		print('number of measurements:', self.N)
		print('dimension of beta:     ', self.k_beta)
		print('dimension of gama:     ', self.k_gama)
		print('dimension of deta:     ', self.k_deta)
		print('cond number of X cov:  ', cx)
		print('cond number of Z cov:  ', cz)
		print('cond number of D cov:  ', cd)

	def checkUPrior(self, v, vname, vsize=None):
		if v is None: return None
		assert v.shape[0]==2, vname+': num of rows must be 2.'
		assert np.all(v[0]<=v[1]),\
			vname+': ubounds must be greater or equal than lbounds.'
		if vsize is not None:
			assert v.shape[1]==vsize, vname+': wrong num of cols.'

	def checkGPrior(self, v, vname, vsize=None):
		if v is None: return None
		assert v.shape[0]==2, vname+': num of rows must be 2.'
		assert np.all(v[1]>0.0), vname+': variance must be positive.'
		if vsize is not None:
			assert v.shape[1]==vsize, vname+': wrong num of cols.'

	# default options
	# -------------------------------------------------------------------------
	def defaultOptions(self):
		# default not using trimming
		self.use_trimming = False
		#
		# default not using bspline
		self.bspline_uprior = None
		self.bspline_gprior = None
		#
		# default not having constrains
		self.constraint_matrix = None
		self.constraint_values = None
		#
		# default solution
		self.soln = None
		self.info = 'no solution, haven\'t run the solver yet.'
		#
		# add default uprior (constrains)
		self.beta_uprior = self.defaultUPrior(self.k_beta)
		self.gama_uprior = self.positiveUPrior(self.k_gama)
		self.deta_uprior = self.positiveUPrior(self.k_deta)
		self.uprior = np.hstack((
			self.beta_uprior,
			self.gama_uprior,
			self.deta_uprior
			))
		#
		self.beta_gprior = self.defaultGPrior(self.k_beta)
		self.gama_gprior = self.defaultGPrior(self.k_gama)
		self.deta_gprior = self.defaultGPrior(self.k_deta)
		self.gprior = np.hstack((
			self.beta_gprior,
			self.gama_gprior,
			self.deta_gprior
			))

	def defaultUPrior(self, col_len):
		uprior = np.empty((2,col_len))
		uprior[0] = -np.inf
		uprior[1] =  np.inf
		#
		return uprior

	def defaultGPrior(self, col_len):
		gprior = np.empty((2,col_len))
		gprior[0] = 0.0
		gprior[1] = np.inf
		#
		return gprior

	def positiveUPrior(self, col_len):
		uprior = np.empty((2,col_len))
		uprior[0] = 1e-8
		uprior[1] = np.inf
		#
		return uprior

	def negativeUPrior(self, col_len):
		uprior = np.empty((2,col_len))
		uprior[0] = -np.inf
		uprior[1] = -1e-8
		#
		return uprior

	# add uniform prior
	# -------------------------------------------------------------------------
	def addUPrior(self, beta_uprior, gama_uprior, deta_uprior):
		# check input
		self.checkUPrior(beta_uprior, 'beta_uprior', vsize=self.k_beta)
		self.checkUPrior(gama_uprior, 'gama_uprior', vsize=self.k_gama)
		self.checkUPrior(deta_uprior, 'deta_uprior', vsize=self.k_deta)
		#
		assert np.all(gama_uprior[0]>=0.0), \
			'gama_uprior: lbounds for variance must be non-negative'
		assert np.all(deta_uprior[0]>=0.0), \
			'deta_uprior: lbounds for variance must be non-negative'
		#
		self.beta_uprior = beta_uprior
		self.gama_uprior = gama_uprior
		self.deta_uprior = deta_uprior
		#
		self.uprior = np.hstack((
			self.beta_uprior,
			self.gama_uprior,
			self.deta_uprior
			))

	# add gaussian prior
	# -------------------------------------------------------------------------
	def addGPrior(self, beta_gprior, gama_gprior):
		# check input
		self.checkGPrior(beta_gprior, 'beta_gprior', vsize=self.k_beta)
		self.checkGPrior(gama_gprior, 'gama_gprior', vsize=self.k_gama)
		self.checkGPrior(deta_gprior, 'deta_gprior', vsize=self.k_deta)
		#
		self.beta_gprior = beta_gprior
		self.gama_gprior = gama_gprior
		self.deta_gprior = deta_gprior
		#
		self.gprior = np.hstack((
			self.beta_gprior,
			self.gama_gprior,
			self.deta_gprior
			))
	
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
		# check the priors of bspline
		self.checkUPrior(bspline_uprior, 'bspline_uprior', vsize=len(knots)-1)
		self.checkUPrior(bspline_gprior, 'bspline_gprior', vsize=len(knots)-1)
		self.bspline_uprior = bspline_uprior
		self.bspline_gprior = bspline_gprior
		#
		# create design matrix
		X_bs, H_bs = dmatrix(self.X[:,col_id], knots, degree,
					l_linear=l_linear, r_linear=r_linear)
		#
		self.k_beta_bs = X_bs.shape[1]
		#
		# constrains: monotonicity
		C_bs = np.array([]).reshape(0, self.k_beta_bs)
		c_bs = np.array([]).reshape(2, 0)
		#
		if bspline_mono is not None:
			C_bs = self.diff1Mat(self.k_beta_bs)
			#
			if bspline_mono == 'increasing':
				c_bs = self.positiveUPrior(self.k_beta_bs - 1)
			if bspline_mono == 'decreasing':
				c_bs = self.negativeUPrior(self.k_beta_bs - 1)
		#
		if bspline_cvcv is not None:
			C_bs = np.vstack((C_bs, self.diff2Mat(self.k_beta_bs)))
			#
			if bspline_cvcv == 'convex':
				c_bs = np.hstack(
					(c_bs, self.positiveUPrior(self.k_beta_bs - 2)))
			if bspline_cvcv == 'concave':
				c_bs = np.hstack(
					(c_bs, self.negativeUPrior(self.k_beta_bs - 2)))
		#
		# constrains: uniform prior on bspline
		if bspline_uprior is not None:
			C_bs = np.vstack((H_bs, C_bs))
			c_bs = np.hstack((bspline_uprior, c_bs))

		# extend x and priors
		self.bsplineExtend(X_bs, H_bs, C_bs, c_bs, col_id)


	def bsplineExtend(self, X_bs, H_bs, C_bs, c_bs, col_id):
		# decide the delete cols id
		del_id = [col_id]
		if np.all(self.X[:,0]==1.0): del_id.append(0)
		#
		# extend the current X cov matrix
		# ---------------------------------------------------------------------
		self.X = self.delThenStack(self.X, X_bs, del_id)
		self.k_beta = self.X.shape[1]
		self.k = self.k_beta + self.k_gama + self.k_deta
		#
		self.id_beta = slice(0, self.k_beta)
		self.id_gama = slice(self.k_beta, self.k_beta + self.k_gama)
		self.id_deta = slice(self.k_beta + self.k_gama, self.k)
		#
		# add to the current beta uprior
		# ---------------------------------------------------------------------
		beta_uprior_bs = self.defaultUPrior(self.k_beta_bs)
		self.beta_uprior =\
			self.delThenStack(self.beta_uprior, beta_uprior_bs, del_id)
		self.uprior = np.hstack((
			self.beta_uprior,
			self.gama_uprior,
			self.deta_uprior
			))
		#
		# add to the current beta gprior
		# ---------------------------------------------------------------------
		beta_gprior_bs = self.defaultGPrior(self.k_beta_bs)
		self.beta_gprior = \
			self.delThenStack(self.beta_gprior, beta_gprior_bs, del_id)
		self.gprior = np.hstack((
			self.beta_gprior,
			self.gama_gprior,
			self.deta_gprior
			))
		#
		# add bspline prior
		# ---------------------------------------------------------------------
		self.H = np.hstack(
			(H_bs, np.zeros((H_bs.shape[0], self.k_beta - self.k_beta_bs))))
		#
		# add bspline constraints
		# ---------------------------------------------------------------------
		if C_bs.size > 0:
			C = np.hstack(
				(C_bs, np.zeros((C_bs.shape[0], self.k - self.k_beta_bs))))
			self.constraint_matrix = C
			self.constraint_values = c_bs

	def diff1Mat(self, k):
		M = np.zeros((k-1,k))
		diag_id0 = np.diag_indices(M.shape[0])
		diag_id1 = (diag_id0[0],diag_id0[1]+1)
		M[diag_id0] = -1.0
		M[diag_id1] =  1.0
		#
		return M

	def diff2Mat(self, k):
		M = np.zeros((k-2,k))
		diag_id0 = np.diag_indices(M.shape[0])
		diag_id1 = (diag_id0[0],diag_id0[1]+1)
		diag_id2 = (diag_id1[0],diag_id1[1]+1)
		M[diag_id0] =  1.0
		M[diag_id1] = -2.0
		M[diag_id2] =  1.0
		#
		return M

	def delThenStack(self, X, Y, del_id):
		X = np.delete(X, del_id, 1)
		return np.hstack((Y, X))

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

	# optimization: the maximum likelihood
	# -------------------------------------------------------------------------
	def optimizeSFA(self, print_level=0, max_iter=50):
		# create problem
		if self.constraint_matrix is None:
			handle = ipopt.problem(
				n=self.k,
				m=0,
				problem_obj=sfaObj(self),
				lb=self.uprior[0],
				ub=self.uprior[1]
				)
		else:
			handle = ipopt.problem(
				n=self.k,
				m=self.constraint_matrix.shape[0],
				problem_obj=sfaObj(self),
				lb=self.uprior[0],
				ub=self.uprior[1],
				cl=self.constraint_values[0],
				cu=self.constraint_values[1]
				)
		# add options
		handle.addOption('print_level', print_level)
		if max_iter is not None: handle.addOption('max_iter', max_iter)
		# initial point
		if self.soln is None:
			beta0 = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y))
			gama0 = np.repeat(0.01, self.k_gama)
			deta0 = np.repeat(0.01, self.k_deta)
			x0 = np.hstack((beta0, gama0, deta0))
		else:
			x0 = self.soln
		# solver the problem
		soln, info = handle.solve(x0)
		# extract the solution
		self.soln = soln
		self.info = info
		self.beta_soln = soln[self.id_beta]
		self.gama_soln = soln[self.id_gama]
		self.deta_soln = soln[self.id_deta]

	# optimization: trimming method on the maximum likelihood
	# -------------------------------------------------------------------------
	def optimizeSFAWithTrimming(self, h, stepsize=1.0, max_iter=100, tol=1e-6,
			verbose=False):
		# add the trimming parameters into object
		self.addTrimming(h)
		#
		# initialize the iteration
		self.optimizeSFA()
		g = self.wGrad(self.soln)
		#
		# start iteration
		err = tol + 1.0
		iter_count = 0
		#
		while err >= tol:
			# proximal gradient step
			w_new = self.wProj(self.w - stepsize*g)
			# update information
			err = np.linalg.norm(w_new - self.w)/stepsize
			iter_count += 1
			#
			np.copyto(self.w, w_new)
			#
			self.optimizeSFA()
			g = self.wGrad(self.soln)
			obj = self.w.dot(g)
			#
			if verbose:
				print('iter %4d, obj %8.2e, err %8.2e' %
					(iter_count, obj, err))
			#
			if iter_count >= max_iter:
				print('trimming reach maximum number of iterations')
				break

	def wGrad(self, x):
		beta = x[self.id_beta]
		gama = x[self.id_gama]
		deta = x[self.id_deta]
		# residual and all variances and stds
		r  = self.Y - self.X.dot(beta)
		#
		vu = self.Z.dot(gama)
		vv = self.D.dot(deta)
		if deta[0] < 0.0: print(deta)
		#
		su = sqrt(vu)
		sv = sqrt(vv)
		#
		v1 = self.V + vu
		v2 = self.V + vu + vv
		#
		if self.vtype == 'half_normal':
			g = 0.5*r**2/v2 + 0.5*log(2.0*pi*v2) - \
				log_erfc(r*sv/sqrt(2.0*v1*v2))
		if self.vtype == 'exponential':
			a = (v1 + r*sv)/sqrt(2.0*v1)
			b = a/sv
			abs_a = sqrt(a*a)
			g = 0.5*r**2/v1 + log(2.0*abs_a) - special(b)
		#
		return g

	def wProj(self, w):
		a = np.min(w) - 1.0
		b = np.max(w) - 0.0
		#
		f = lambda x:\
		    np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - self.h
		#
		x = bisect(f, a, b)
		#
		return np.maximum(np.minimum(w - x, 1.0), 0.0)

	# post analysis
	# -------------------------------------------------------------------------
	def estimateRE(self):
		r  = self.X.dot(self.beta_soln) - self.Y
		vu = self.Z.dot(self.gama_soln)
		vv = self.D.dot(self.deta_soln)
		#
		if self.vtype == 'half_normal':
			self.v_soln = np.maximum(0.0, vv*r/(self.V + vv + vu))
		if self.vtype == 'exponential':
			self.v_soln = np.maximum(0.0, r - (self.V + vu)/np.sqrt(vv))
		#
		self.u_soln = vu*(self.v_soln - r)/(self.V + vu)
		

# IPOPT objective: maximum likelihood
# =============================================================================
class sfaObj:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, sfa):
		self.sfa = sfa
		#
		if sfa.constraint_matrix is not None:
			C = sfa.constraint_matrix
			self.constraints = lambda x: C.dot(x)
			self.jacobian = lambda x: C.reshape(C.size)

	# objective
	# -------------------------------------------------------------------------
	def objective(self, x):
		sfa = self.sfa
		# likelihood
		g = sfa.wGrad(x)
		#
		# trimming
		# ---------------------------------------------------------------------
		if sfa.use_trimming:
			f = sfa.w.dot(g)
		else:
			f = np.sum(g)
		#
		# add priors
		# ---------------------------------------------------------------------
		f += np.sum( (x - sfa.gprior[0])**2/sfa.gprior[1] )
		#
		# add bspline priors
		# ---------------------------------------------------------------------
		if sfa.bspline_gprior is not None:
			f += np.sum( (mr.H.dot(beta) - mr.bspline_gprior[0])**2 /
					mr.bspline_gprior[1] )
		#
		return f

	# gradient
	# -------------------------------------------------------------------------
	def gradient(self, x, eps=1e-10):
		g = np.empty(x.size)
		z = x + 0j
		#
		for i in range(x.size):
			z[i] += 1j*eps
			f     = self.objective(z)
			g[i]  = f.imag/eps
			z[i] -= 1j*eps
		#
		return g
