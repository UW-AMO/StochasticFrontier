# src file of the pysfa class

import numpy as np

from numpy             import exp, log, sqrt, pi
from numpy.linalg      import det, norm, solve
from scipy.special     import erf, erfc
from ipopt             import minimize_ipopt
from sfa_utils.npufunc import log_erfc


class SFA:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, m, x, s, vtype='hnl'):
		# test data consistancy
		assert x.shape[0] == m, 'x inconsistant with number of mea'
		assert s.size == m, 'mea stds inconsistant with number of mea'
		assert det(x.T.dot(x)) > 1e-10,\
			'model mis-specification, x has redundancy (is singular)'
		# construct the object
		self.m = m
		self.x = x
		self.s = s
		self.k = x.shape[1]
		self.vtype = vtype
		# initialize the uniform prior on variables
		self.beta_uniform_prior = np.array([[-np.inf, np.inf]]*self.k)
		self.su_uniform_prior = np.array([0.0, np.inf])
		self.sv_uniform_prior = np.array([0.0, np.inf])
		
	# data simulation
	# -------------------------------------------------------------------------
	def simData(self, beta_t, su_t, sv_t):
		assert beta_t.size == self.k, 'wrong size of true beta'
		assert isinstance(su_t, float), 'true su need to be float scalar'
		assert isinstance(sv_t, float), 'true sv need to be float scalar'
		#
		self.beta_t = beta_t
		self.su_t   = su_t
		self.sv_t   = sv_t
		#
		self.eps = np.random.randn(self.m)*self.s
		self.u   = np.random.randn(self.m)*self.su_t
		if self.vtype == 'hnl':
			self.v = np.abs(np.random.randn(self.m)*self.sv_t)
		elif self.vtype == 'exp':
			self.v = np.random.exponential(scale=self.sv_t, size=self.m)
		self.y   = self.x.dot(self.beta_t) + self.u - self.v + self.eps

	# maximum likelihood objective function
	# -------------------------------------------------------------------------
	def maxlFunc(self, beta, su, sv):
		# residual
		r = self.y - self.x.dot(beta)
		# variances
		v_su  = self.s**2 + su**2
		v_suv = self.s**2 + su**2 + sv**2
		#
		if self.vtype == 'hnl':
			return np.sum(
				0.5*r**2/v_suv + 0.5*log(2.0*pi*v_suv) -
				log_erfc(sv*r/sqrt(2.0*v_su*v_suv))
				)
		elif self.vtype == 'exp':
			a = (v_su + sv*r)/sqrt(2.0*v_su)
			# f = np.sum(-0.5*(v_su + 2.0*sv*r)/sv**2 - log(0.5*erfc(a/sv)/sv))
			f = 0.0
			if sv < 1e-2:
				return np.sum(
					0.5*r**2/v_su + 0.5*log(4.0*pi*a**2) -\
					log(1.0-0.5*sv**2/a**2)
					)
			else:
				return np.sum(
					-0.5*(v_su + 2.0*sv*r)/sv**2 + log(2.0*sv) -\
					log_erfc(a/sv)
					)
			return f

	# maximum likelihood objective function for solver
	# -------------------------------------------------------------------------
	def solver_maxlFunc(self, z):
		# extract variables
		beta = z[:self.k]
		su   = z[-2]
		sv   = z[-1]
		#
		return self.maxlFunc(beta, su, sv)

	# maximum likelihood gradient function for solver
	# -------------------------------------------------------------------------
	def solver_maxlGrad(self, z, eps=1e-10):
		grad = np.empty(z.shape)
		z_c  = z.copy() + 0j
		#
		for i in range(z.size):
			z_c[i] += 1j*eps
			#
			func = self.solver_maxlFunc(z_c)
			grad[i] = func.imag/eps
			#
			z_c[i] -= 1j*eps
		#
		return grad

	# optimize the maximum likelihood
	# -------------------------------------------------------------------------
	def fitMaxl(self):
		# initialize the parameters
		beta = solve(self.x.T.dot(self.x), self.x.T.dot(self.y))
		su   = 0.0
		sv   = 0.0
		z0   = np.hstack((beta, su, sv))
		# non-negative constrain on su and sv
		bounds = np.vstack((
			self.beta_uniform_prior,
			self.su_uniform_prior,
			self.sv_uniform_prior
			))
		# apply ipopt solver
		res = minimize_ipopt(
			self.solver_maxlFunc,
			z0,
			jac=self.solver_maxlGrad,
			bounds=bounds,
			options={'print_level': 0}
			)
		# extract the solution
		self.beta_soln = res.x[:self.k]
		self.su_soln   = res.x[-2]
		self.sv_soln   = res.x[-1]
