import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/')
from pysfa import SFA


# generate data
# -----------------------------------------------------------------------------
np.random.seed(123)
m = 1000
k = 2
x = np.random.randn(m,k)
s = np.ones(m)*0.1
#
beta_t = np.random.randn(k)
su_t = 0.1
sv_t = 0.1

# create objec
# -----------------------------------------------------------------------------
sfa = SFA(m, x, s, vtype='hnl')
sfa.simData(beta_t, su_t, sv_t)

# apply solver
# -----------------------------------------------------------------------------
sfa.fitMaxl()
print('beta_t:', beta_t, ', beta_soln:', sfa.beta_soln)
print('su_t:', su_t, ', beta_soln:', sfa.su_soln)
print('sv_t:', sv_t, ', beta_soln:', sfa.sv_soln)