from jax import random
import jax.numpy as jnp

####################
### Set RNG seed ###
####################

class PRNG(object):
    def __init__(self, seed):
        self.key= random.PRNGKey(seed)

    def next(self):
        k1, k2 = random.split(self.key)
        self.key = k1
        return k2

#################
### Constants ###
#################

SAFE_VALUE = 0.03 #0.1
UNSAFE_VALUE = -0.03 #-0.3
GAMMA = 0.01 #0.01
RNG = PRNG(854721)
N_RANDOM_FEATURES = 8 #100
X_DIM = 2
U_DIM = 2
SIGMA = 0.66 #0.66
PSI = 1.0
RF_WEIGHTS = (random.normal(RNG.next(), shape=(X_DIM, N_RANDOM_FEATURES))*SIGMA, 
              random.uniform(RNG.next(), minval=0, maxval=2*jnp.pi, shape=(N_RANDOM_FEATURES,)))

CONSTANTS = {
        "SAFE_VALUE" : SAFE_VALUE,
        "UNSAFE_VALUE" : UNSAFE_VALUE,
        "GAMMA" : GAMMA,
        "RNG" : RNG,
        "N_RANDOM_FEATURES" : N_RANDOM_FEATURES,
        "X_DIM" : X_DIM,
        "U_DIM" : U_DIM,
        "SIGMA" : SIGMA,
        "PSI" : PSI,
        "RF_WEIGHTS" : RF_WEIGHTS,
        }

def GET_CONSTANTS(name):
    return CONSTANTS[name] 

