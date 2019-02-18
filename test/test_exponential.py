import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/')
from pysfa import SFA


# generate data
# -----------------------------------------------------------------------------
np.random.seed(123)
m = 2000
k_beta = 2
k_gama = 1
k_deta = 1
x = np.random.randn(m,k_beta)
z = np.ones((m,k_gama))
d = np.ones((m,k_deta))
s = np.ones(m)*0.1
#
beta_t = np.random.randn(k_beta)
gama_t = np.random.rand(k_gama)
deta_t = np.random.rand(k_deta)

# create objec
# -----------------------------------------------------------------------------
sfa = SFA(x, z, d, s, vtype='exponential')
sfa.simData(beta_t, gama_t, deta_t)

# apply solver
# -----------------------------------------------------------------------------
sfa.optimizeSFA()
print('beta_t:', beta_t, ', beta_soln:', sfa.beta_soln)
print('gama_t:', gama_t, ', gama_soln:', sfa.gama_soln)
print('deta_t:', deta_t, ', deta_soln:', sfa.deta_soln)