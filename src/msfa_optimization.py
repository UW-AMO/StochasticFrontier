import numpy as np
import ipopt

# optimization: the maximum likelihood
# -------------------------------------------------------------------------
def optimizeSFA(msfa, print_level=0, max_iter=100):
	# create problem
	if not msfa.use_constraint:
		handle = ipopt.problem(
			n=msfa.k,
			m=0,
			problem_obj=msfaObj(msfa),
			lb=msfa.uprior[0],
			ub=msfa.uprior[1]
			)
	else:
		handle = ipopt.problem(
			n=msfa.k,
			m=msfa.num_constraint,
			problem_obj=msfaObj(msfa),
			lb=msfa.uprior[0],
			ub=msfa.uprior[1],
			cl=msfa.constraint_values[0],
			cu=msfa.constraint_values[1]
			)
	# add options
	handle.addOption('print_level', print_level)
	if max_iter is not None: handle.addOption('max_iter', max_iter)
	# initial point
	if msfa.soln is None:
		beta0 = np.linalg.solve(msfa.X.T.dot(msfa.X), msfa.X.T.dot(msfa.Y))
		gama0 = np.repeat(0.01, msfa.k_gama)
		deta0 = np.repeat(0.01, msfa.k_deta)
		x0 = np.hstack((beta0, gama0, deta0))
	else:
		x0 = msfa.soln
	# solver the problem
	soln, info = handle.solve(x0)
	# extract the solution
	msfa.soln = soln
	msfa.info = info
	msfa.beta_soln = soln[msfa.id_beta]
	msfa.gama_soln = soln[msfa.id_gama]
	msfa.deta_soln = soln[msfa.id_deta]

# optimization: trimming method on the maximum likelihood
# -------------------------------------------------------------------------
def optimizeSFAWithTrimming(msfa, h, stepsize=1.0, max_iter=100, tol=1e-6,
		verbose=False):
	# add the trimming parameters into object
	msfa.addTrimming(h)
	#
	# initialize the iteration
	optimizeSFA(msfa)
	g = wGrad(msfa.soln)
	#
	# start iteration
	err = tol + 1.0
	iter_count = 0
	#
	while err >= tol:
		# proximal gradient step
		w_new = wProj(msfa.w - stepsize*g)
		# update information
		err = np.linalg.norm(w_new - msfa.w)/stepsize
		iter_count += 1
		#
		np.copyto(msfa.w, w_new)
		#
		optimizeSFA(msfa)
		g = wGrad(msfa.soln)
		obj = msfa.w.dot(g)
		#
		if verbose:
			print('iter %4d, obj %8.2e, err %8.2e' %
				(iter_count, obj, err))
		#
		if iter_count >= max_iter:
			print('trimming reach maximum number of iterations')
			break

def wGrad(msfa, x):
	return msfa.likelihood(x)

def wProj(self, w):
	a = np.min(w) - 1.0
	b = np.max(w) - 0.0
	#
	f = lambda x:\
	    np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - msfa.h
	#
	x = bisect(f, a, b)
	#
	return np.maximum(np.minimum(w - x, 1.0), 0.0)

# post analysis
# -------------------------------------------------------------------------
def estimateRE(msfa):
	r  = msfa.X.dot(msfa.beta_soln) - msfa.Y
	vu = msfa.Z.dot(msfa.gama_soln)
	vv = msfa.D.dot(msfa.deta_soln)
	#
	if msfa.vtype == 'half_normal':
		msfa.v_soln = np.maximum(0.0, vv*r/(msfa.V + vv + vu))
	if msfa.vtype == 'exponential':
		msfa.v_soln = np.maximum(0.0, r - (msfa.V + vu)/np.sqrt(vv))
	#
	msfa.u_soln = vu*(msfa.v_soln - r)/(msfa.V + vu)
	

# IPOPT objective: maximum likelihood
# =============================================================================
class msfaObj:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, msfa):
		self.msfa = msfa
		#
		if msfa.use_constraint:
			C = msfa.constraint_matrix
			self.constraints = lambda x: C.dot(x)
			self.jacobian = lambda x: C

	# objective
	# -------------------------------------------------------------------------
	def objective(self, x):
		return self.msfa.objective(x)

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