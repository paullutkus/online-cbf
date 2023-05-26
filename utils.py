import numpy as np
import jax.numpy as jnp
from CONSTANTS import GET_CONSTANTS

#################
### Constants ###
#################

RF_WEIGHTS = GET_CONSTANTS("RF_WEIGHTS")

############
### Misc ###
############

def binary_search(d, x, h, params, bias_param, eps=1e-3, scale=2):
    while jnp.abs(h(x, params, bias_param, RF_WEIGHTS)) > eps:
        if h(x, params, bias_param, RF_WEIGHTS) > 0:
            x_prime = x + scale * d
            if h(x_prime, params, bias_param, RF_WEIGHTS) < 0:
                scale /= 2
        else:
            x_prime = x - scale * d
            if h(x_prime, params, bias_param, RF_WEIGHTS) > 0:
                scale /= 2
        x = x_prime
    return x 

