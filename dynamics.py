import numpy as np
import jax
import jax.numpy as jnp
import scipy
import cvxpy as cp
from CONSTANTS import GET_CONSTANTS

#################
### Constants ###
#################

DELTA = 1.0
RF_WEIGHTS = GET_CONSTANTS("RF_WEIGHTS")

################
### Dynamics ###
################

def dynamics_f(x):
    return -x

def dynamics_g(x):
    return jnp.diag(x**2 + DELTA)

def dynamics(x, u):
    # Alternative, unambiguous dynamics)
    return jnp.matmul(jnp.diag(-jnp.ones(2)), x) + jnp.matmul(jnp.diag(jnp.array([x[0]**2+DELTA, x[1]**2+DELTA])), u)
    # Concise, ambiguous dynamics
    #return -x + (x**2 + DELTA) * u



