import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, device_get, device_put, jacobian, jacfwd, jacrev
import casadi
import cvxpy as cp
from dynamics import dynamics_f, dynamics_g
from CONSTANTS import GET_CONSTANTS

#################
### CONSTANTS ###
#################

RF_WEIGHTS = GET_CONSTANTS("RF_WEIGHTS")
GAMMA = GET_CONSTANTS("GAMMA")
N_RANDOM_FEATURES = GET_CONSTANTS("N_RANDOM_FEATURES")
PSI = GET_CONSTANTS("PSI")


#############
### Model ###
#############

def phi(X, rf_weights):
    W, b = rf_weights
    n_rf = len(b)
    return jnp.sqrt(2.0 / n_rf) * jnp.cos(X.dot(W) + b)

def h_model(x, theta, bias, rf_weights):
    return jnp.max(jnp.matmul(phi(x, rf_weights), jnp.squeeze(jnp.array(theta)).T) + jnp.array(bias))

###################
### Solve Model ###
###################

def solve_model(data, gamma_xu_constraints, gamma_xu_fillers, safe_values, unsafe_values, verbose=False, use_bias=True, norm_trick=False, use_casadi=True):
    x_constraint, u_constraint, x_filler, u_filler, unsafe_points, unsafe_inputs = data
    assert len(gamma_xu_constraints) == x_constraint.shape[0]
    assert (gamma_xu_constraints >= 0).all()
    assert len(gamma_xu_fillers) == x_filler.shape[0]
    assert (gamma_xu_fillers >= 0).all()
    assert len(safe_values) == x_constraint.shape[0]
    assert (safe_values >= 0).all()
    assert len(unsafe_values) == unsafe_points.shape[0]
    assert (unsafe_values <= 0).all()

    if use_casadi:
        opti = casadi.Opti()

    #assert not n_safe_samples

    if use_casadi:
        theta = opti.variable(N_RANDOM_FEATURES)
        bias = opti.variable()
        param_cost = casadi.sumsqr(theta) + bias ** 2
    else:
        theta = cp.Variable(N_RANDOM_FEATURES)
        bias = cp.Variable()
        param_cost = cp.sum_squares(theta) + bias ** 2
    lam_safe = 1

    constraints = []

    if not use_bias:
        constraints.append(bias == 0)

    if norm_trick:
        print("using norm trick")
        phis = device_get(phi(x_constraint, RF_WEIGHTS))
        Dphixdots = device_get(
                vmap(lambda x: jnp.dot(jacobian(phi, argnums=0)(x, RF_WEIGHTS), dynamics_f(x)),
                    in_axes=0)(x_constraint))
        Dz = device_get(
                vmap(lambda x: jacobian(phi, argnums=0)(x, RF_WEIGHTS) @ dynamics_g(x).T,
                    in_axes=0)(x_constraint))
        for this_phi, this_Dphixdot, this_Dz, this_gamma in zip(phis, Dphixdots, Dz, gamma_xu_constraints):
            if use_casadi:
                constraints.append((theta.T @ (this_Dphixdot + PSI * this_phi) + casadi.norm_2(theta.T @ this_Dz) + PSI * bias) >= this_gamma)
            else:
                constraints.append((theta.T * (this_Dphixdot + PSI * this_phi) + cp.norm(theta.T * this_Dz, 1) + PSI * bias) >= this_gamma)
    else:
        phis = device_get(phi(x_constraint, RF_WEIGHTS))
        Dphixdots = device_get(
            vmap(lambda x, u: jnp.dot(jacobian(phi, argnums=0)(x, RF_WEIGHTS), dynamics(x, u)),
                 in_axes=(0, 0))(x_constraint, u_constraint))

        for this_phi, this_Dphixdot, this_gamma in zip(phis, Dphixdots, gamma_xu_constraints):
            if use_casadi:
                constraints.append((theta.T @ (this_Dphixdot + PSI * this_phi) + PSI * bias) >= this_gamma)
            else:
                constraints.append((theta.T * (this_Dphixdot + PSI * this_phi) + PSI * bias) >= this_gamma)

    if norm_trick:
        print("using norm trick")
        phis = device_get(phi(x_filler, RF_WEIGHTS))
        Dphixdots = device_get(
                vmap(lambda x: jnp.dot(jacobian(phi, argnums=0)(x, RF_WEIGHTS), dynamics_f(x)),
                    in_axes=0)(x_filler))
        Dz = device_get(
                vmap(lambda x: jacobian(phi, argnums=0)(x, RF_WEIGHTS) @ dynamics_g(x).T,
                    in_axes=0)(x_filler))
        for this_phi, this_Dphixdot, this_Dz, this_gamma in zip(phis, Dphixdots, Dz, gamma_xu_fillers):
            if use_casadi:
                constraints.append((theta.T @ (this_Dphixdot + PSI * this_phi) + casadi.norm_2(theta.T @ this_Dz) + PSI * bias) >= this_gamma)
            else:
                constraints.append((theta.T * (this_Dphixdot + PSI * this_phi) + cp.norm(theta.T * this_Dz, 1) + PSI * bias) >= this_gamma)
    else:
        phis = device_get(phi(x_filler, RF_WEIGHTS))
        Dphixdots = device_get(
            vmap(lambda x, u: jnp.dot(jacobian(phi, argnums=0)(x, RF_WEIGHTS), dynamics(x, u)),
                 in_axes=(0,0))(x_filler, u_filler))

        # Consider not enforcing this condition
        # Increase features?
        for this_phi, this_Dphixdot, this_gamma in zip(phis, Dphixdots, gamma_xu_fillers):
            if use_casadi:
                constraints.append((theta.T @ (this_Dphixdot + PSI * this_phi) + PSI * bias) >= this_gamma)
            else:
                constraints.append((theta.T * (this_Dphixdot + PSI * this_phi) + PSI * bias) >= this_gamma)

    safe_cost = 0
    phis = device_get(phi(x_constraint, RF_WEIGHTS))
    for this_phi, this_safe_value in zip(phis, safe_values):
        if use_casadi:
            constraints.append((theta.T @ this_phi + bias) >= this_safe_value)
            safe_cost += (theta.T @ this_phi + bias - lam_safe) ** 2
        else:
            constraints.append((theta.T * this_phi + bias) >= this_safe_value)
            safe_cost += (theta.T * this_phi + bias - lam_safe) ** 2

  # What is this for?
    '''
    if n_safe_samples:
        phis = device_get(phi(safe_points, RF_WEIGHTS))
        for this_phi in phis:
            constraints.append((theta.T * this_phi + bias) >= safe_value)
            safe_cost += (theta.T * this_phi + bias - lam_safe) ** 2
    '''

    unsafe_cost = 0
    phis = device_get(phi(unsafe_points, RF_WEIGHTS))
    for this_phi, this_unsafe_value in zip(phis, unsafe_values):
        if use_casadi:
            constraints.append((theta.T @ this_phi + bias) <= this_unsafe_value)
            unsafe_cost += (theta.T @ this_phi + bias + lam_safe) ** 2
        else:
            constraints.append((theta.T * this_phi + bias) <= this_unsafe_value)
            unsafe_cost += (theta.T * this_phi + bias + lam_safe) ** 2


    x_all = np.vstack((x_constraint, unsafe_points))
    phis = device_get(phi(x_all, RF_WEIGHTS))
    Dphis = device_get(
        vmap(lambda x: jacobian(phi, argnums=0)(x, RF_WEIGHTS), in_axes=(0))(x_all))

    Dh_cost = 0
    for this_Dphi in Dphis:
        if use_casadi:
            Dh_cost += casadi.sumsqr(theta.T @ this_Dphi)
        else:
            Dh_cost += cp.sum_squares(theta.T @ this_Dphi)

    if use_casadi:
        opti.minimize(param_cost + Dh_cost)
        opti.subject_to(constraints)
        opti.set_initial(theta, casadi.DM.rand(N_RANDOM_FEATURES))
        p_opts = {"expand" : True, "verbose" : False, "print_time" : verbose}
        s_opts = {"max_iter" : 1000, "print_level" : 5*int(verbose)}
        opti.solver('ipopt', p_opts, s_opts)
        sol = opti.solve()
        return sol.value(theta), sol.value(bias)

    else:
        obj = cp.Minimize(param_cost + 0 * safe_cost + 0 * unsafe_cost + Dh_cost)

        params = None
        bias_param = None
        prob = cp.Problem(obj, constraints)
        solver = cp.MOSEK
        if solver == cp.MOSEK:
            import mosek
            mosek.dparam.basis_rel_tol_s=1e-2
            mosek.dparam.basis_tol_s=1e4
            mosek.dparam.basis_rel_tol=1e-2
            mosek.dparam.basis_tol_x=1e4

        result = prob.solve(solver=solver, verbose=verbose)
        assert prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        params = device_put(theta.value)
        bias_param = device_put(bias.value)
        return params, bias_param

################
### Plot CBF ###
################

def plot_cbf(h_model, params, bias_param, traj=None): 
    x1 = jnp.linspace(-2, 2, num=50)
    x2 = jnp.linspace(-2, 2, num=50) 
    hvals = vmap(lambda s1: vmap(lambda s2: h_model(jnp.array([s1, s2]), params, bias_param, RF_WEIGHTS))(x2))(x1)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(6, 6))
    contour_plot = plt.contour(jnp.linspace(-2, 2, num=50), jnp.linspace(-2, 2, num=50), hvals.T)
    plt.clabel(contour_plot, inline=1, fontsize=10)
    plt.plot(jnp.linspace(-2, 1, num=20), np.ones((20,)), 'k:', linewidth=2)
    plt.plot(jnp.array([1, 1]), jnp.array([-2, 1]), 'k:', linewidth=1.5)
    if traj is not None:
        plt.plot(traj[:,0], traj[:,1], 'r.')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    x1 = jnp.linspace(-2, 2, num=30)
    x2 = jnp.linspace(-2, 2, num=30)
    xx, yy = jnp.meshgrid(x1, x2)
    zz = vmap(lambda arg1, arg2: vmap(lambda s1, s2: h_model(jnp.array([s1, s2]), params, bias_param, RF_WEIGHTS), in_axes=(0, 0))(arg1, arg2), in_axes=(0,0))(xx, yy)
    fig = plt.figure(figsize=(16, 10))
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=-30)
    ax.plot_surface(xx, yy, zz)
    plt.show()
    return


