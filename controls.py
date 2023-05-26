import numpy as np
import jax
import jax.numpy as jnp
import scipy
import cvxpy as cp
from CONSTANTS import GET_CONSTANTS
from dynamics import dynamics, dynamics_f, dynamics_g

#################
### Constants ###
#################

DELTA = 1.0
RF_WEIGHTS = GET_CONSTANTS("RF_WEIGHTS")

##############################################################
### Circular reference trajectory & feedback linearization ###
##############################################################

def get_xref(r, origin=(0, 0)):
    assert r >= 0
    def xref(t):
        return jnp.array([-r*jnp.cos(t) + origin[0], r*jnp.sin(t) + origin[1]])
    return xref

def get_feedback_linear_controller(xref, k):
    assert k > 0
    Dxref = jax.jacobian(xref)
    def controller(x, t):
         return 1 / (x**2 + DELTA) * (Dxref(t) + x - k * (x - xref(t)))
    return controller

####################################################
### Linear reference trajectory and adaptive MPC ###
####################################################

def origin_border_xref(pos, steps=20):
    # current position to origin (first_leg)
    first_leg = np.hstack((np.linspace(1, 0, steps//2).reshape(-1, 1), np.linspace(1, 0, steps//2).reshape(-1, 1)))
    first_leg[:,0] *= pos[0]; first_leg[:,1] *= pos[1]

    # origin to random point on unit circle (second leg)
    theta = 2*np.pi*np.random.rand()
    x, y = np.cos(theta), np.sin(theta)
    second_leg = np.hstack((np.linspace(0, 1, steps//2).reshape(-1, 1), np.linspace(0, 1, steps//2).reshape(-1, 1)))
    second_leg[:,0] *= x; second_leg[:,1] *= y

    if steps % 2 == 1:
        xref = np.vstack((first_leg, (first_leg[-1,:] + second_leg[0,:])/2, second_leg))
    else:
        xref = np.vstack((first_leg, second_leg))
    return xref

def endpt_mpc(x, t, x_goal, t_end, steps, horizon, k):
    if t == t_end:
        return np.float32(np.array([0, 0]))
    ### CONSTANT ###
    FINAL_STATE_COST = 10
    CONTROL_COST = 1
    DELTA_T = t_end / steps
    A = jax.jacfwd(dynamics, argnums=0)(x, jnp.array([0.,0.]))
    B = jax.jacfwd(dynamics, argnums=1)(x, jnp.array([0.,0.]))
    if k + horizon <= steps:
        N = horizon
    else:
        N = steps - k
    x_var = cp.Variable(2*N)
    u_var = cp.Variable(2*N) 
    x_0 = x
    x_e = x_var - jnp.tile(x_goal, N)
    obj = cp.quad_form(x_e, FINAL_STATE_COST*jnp.eye(2*N)) + cp.quad_form(u_var, CONTROL_COST*jnp.eye(2*N))
    ### Alternate objective form where final state is penalized more ###
    #obj = cp.quad_form(x_e, jnp.diag(jnp.hstack((jnp.ones((2 * horizon - 2)), jnp.array([FINAL_STATE_COST, FINAL_STATE_COST]))))) \
    #    + cp.quad_form(u_var, CONTROL_COST*jnp.eye(2*horizon))
    init_cons = [x_var[0:2] == x_0]
    cons = init_cons + [x_var[2*k:2*k+2] + DELTA_T * A @ x_var[2*k:2*k+2] + DELTA_T * B @ u_var[2*k:2*k+2] == x_var[2*(k+1):2*(k+1)+2] for k in range(N-1)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=False)
    u_init = u_var.value[:2]
    return np.float32(u_init)

def endpt_mpc_callback(x, t, x_goal, t_end, steps, horizon, k):
    x = x.astype(jnp.result_type(float, x.dtype))
    x_goal = x_goal.astype(jnp.result_type(float, x_goal.dtype))
    x_goal= jnp.asarray(x_goal) 
    u_shape = (2, )
    result_shape = jax.core.ShapedArray(u_shape, x.dtype)
    return jax.pure_callback(endpt_mpc, result_shape, x, t, x_goal, t_end, steps, horizon, k)

def mpc_qp(x, t, xref, t_end, horizon):
    if t == t_end:
        return np.float32(np.array([0, 0]))
    steps = xref.shape[0]
    delta_t = t_end / steps
    k = int(t // delta_t)
    if k + horizon < steps:
        xref_k = xref[k:k+horizon]
    else:
        xref_k = xref[k:]
    dim = xref_k.shape[1]
    x_0 = x
    uref_k = jnp.zeros_like(xref_k) 
    A = jax.jacfwd(dynamics, argnums=0)
    B = jax.jacfwd(dynamics, argnums=1)
    A_map = jax.vmap(A, in_axes=(0, 0), out_axes=0)
    B_map = jax.vmap(B, in_axes=(0, 0), out_axes=0) 
    A_along_xref = A_map(xref_k, uref_k)
    B_along_uref = B_map(xref_k, uref_k)
    x_var = cp.Variable(xref_k.shape[0]*dim)
    u_var = cp.Variable(xref_k.shape[0]*dim)
    x_e = x_var - xref_k.reshape(-1)
    obj = (1 / delta_t) * cp.quad_form(x_e, jnp.eye(xref_k.shape[0]*dim)) + cp.quad_form(u_var, jnp.eye(uref_k.shape[0]*dim))
    init_cons = [x_var[0] == x_0]
    cons = init_cons + [A_along_xref[i] @ x_var[i*2:i*2+2] + B_along_uref[i] @ u_var[i*2:i*2+2] == x_var[(i+1)*2:(i+1)*2+2] for i in range(xref_k.shape[0]-1)]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(verbose=False)
    u_init = np.float32(u_var.value[:2])
    return u_init

def mpc_qp_callback(x, t, xref, t_end, horizon):
    x = x.astype(jnp.result_type(float, x.dtype))
    t = t.astype(jnp.result_type(float, t.dtype))
    xref = xref.astype(jnp.result_type(float, xref.dtype))
    xref = jnp.asarray(xref) 
    u_shape = (2,)
    result_shape = jax.core.ShapedArray(u_shape, x.dtype)
    return jax.pure_callback(mpc_qp, result_shape, x, t, xref, t_end, horizon)
 
def get_adaptive_mpc(xref, t_end, horizon):
    def controller(x, t):
        u = mpc_qp_callback(x, t, xref, t_end, horizon)
        return u
    return controller

class endpt_MPC:
    def __init__(self, x_goal, t_end, steps, horizon):
        self.x_goal = x_goal
        self.t_end = t_end
        self.steps = steps
        self.horizon = horizon

        ### FOR JAX JIT ###
        #self.u = jnp.array(steps * [None])

        self.u = (steps+1) * [None]

    def controller(self, x, t):
        ### FOR JAX JIT ###
        #k = (t // (self.t_end / self.steps)).astype(int)

        k = int(t // (self.t_end / self.steps))

        ### FOR JAX JIT ###
        #if not jnp.isnan(self.u[k]):

        if self.u[k] is not None:
            return self.u[k]
        else:
            self.u[k] = endpt_mpc_callback(x, t, self.x_goal, self.t_end, self.steps, self.horizon, k)
            return self.u[k]

#######################
### Safe controller ###
#######################

def get_safe_controller(r, k, h, params, bias_param, track_control=False, 
                        mpc=False, endpt_mpc=False, pos=None, steps=None, t_end=None, x_goal=None, 
                        horizon=5):
    if endpt_mpc:
        assert((track_control == False) and (x_goal is not None) and (steps is not None) and (t_end is not None))
        mpc = endpt_MPC(x_goal, t_end, steps, horizon)
        opt_controller = mpc.controller

        ### FOR JAX JIT ###
        #jax.jit(mpc.controller)

    elif mpc:
        assert((track_control==False) and (pos is not None) and (steps is not None) and (t_end is not None))
        opt_controller = jax.jit(get_adaptive_mpc(origin_border_xref(pos, steps), t_end, horizon))
    else:
        opt_controller = jax.jit(get_feedback_linear_controller(get_xref(r), k))
    Dh = jax.jit(jax.grad(h, argnums=0))
    def safe_controller(x, t, track_control=track_control):
        Dhx = jax.device_get(Dh(x, params, bias_param, RF_WEIGHTS))
        if track_control:
            tangential = np.array([-Dhx[1], Dhx[0]])
            u_opt = -1e0*(0.75 * Dhx + 0.25 * tangential)
        else:
            u_opt = jax.device_get(opt_controller(x, t))
        f = jax.device_get(dynamics_f(x))
        g = jax.device_get(dynamics_g(x))
        hx = jax.device_get(h(x, params, bias_param, RF_WEIGHTS))
        x = jax.device_get(x)
        u_mod = cp.Variable(len(x))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_opt))
        constraints = [np.dot(Dhx, f) + u_mod.T @ np.dot(g.T, Dhx) + hx >= 0]
        prob = cp.Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS, verbose=False)
        if prob.status != cp.OPTIMAL:
            print("WARNING: problem status " + str(prob.status))
        return u_mod.value
    return safe_controller



