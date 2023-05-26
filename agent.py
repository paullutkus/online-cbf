import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.stats import rankdata
from optim import solve_model, plot_cbf, h_model
from data import kd_tree_detection, mesh_controls
from dynamics import dynamics
from controls import get_safe_controller
from utils import binary_search
from CONSTANTS import GET_CONSTANTS

#################
### Constants ###
#################

SAFE_VALUE = GET_CONSTANTS("SAFE_VALUE")
UNSAFE_VALUE = GET_CONSTANTS("UNSAFE_VALUE")
GAMMA = GET_CONSTANTS("GAMMA")
RF_WEIGHTS = GET_CONSTANTS("RF_WEIGHTS")

#############
### Agent ###
#############

class Agent:
    def __init__(self, data, obs_dict=None, params_and_bias=None, pos=(0,0), t=0, sensor_range=0.25, verbose=False):

        # add dummy control inputs for points
        if type(data) is np.ndarray:
            data = ((data,) + (np.zeros_like(data),),)
            
        if obs_dict is None:
            obs_dict = {}
            # add a dictionary with no obstacles for each collection of points, | combines dictionaries
            for D in data:
                X = D[0]
                obs_dict = obs_dict | {tuple(x) : 0 for x in X} 

        self.data = data
        self.obs_dict = obs_dict
        self.seen_dict    = {pt:0 for pt in self.obs_dict}
        self.learned_dict = {pt:0 for pt in self.obs_dict}
        self.x_pts = np.vstack([d[0] for d in data])
        self.pos = jnp.array(pos)
        self.t = t
        self.sensor_range = sensor_range
        self.h = h_model
        self.verbose=verbose
        if params_and_bias is None:
            self.params = []
            self.bias_param = []
            self.scan(plot=True)
            self.train()
        else:
            self.params, self.bias_param = params_and_bias
     
    def scan(self, plot=False):
        print("position is", self.pos)
        print("time is", self.t)
        x_scan = []
        u_scan = []
        x_scan_unsafe = []
        u_scan_unsafe = []
        for d in self.data:
            for x_i, u_i in zip(*d):
                if np.linalg.norm(self.pos - x_i) <= self.sensor_range:                     
                    if (self.obs_dict[tuple(x_i)] == 0):
                        x_scan.append(x_i)
                        u_scan.append(u_i)
                        self.seen_dict[tuple(np.array(self.pos))] = 1
                    else:
                        x_scan_unsafe.append(x_i)
                        u_scan_unsafe.append(u_i)
        x_scan = np.array(x_scan); u_scan = np.array(u_scan)
        x_scan_unsafe = np.array(x_scan_unsafe); u_scan_unsafe = np.array(u_scan_unsafe)
        color_dict = {0:"yellow",  1:"red",  2:"green",  3:"red",  4:"blue"}
        if plot:
            plt.figure(figsize=(9,9))
            for x in self.x_pts:
                color = color_dict[self.obs_dict[tuple(x)]]
                plt.plot(x[0], x[1], color=color, marker=".", linestyle="none")
            plt.plot(x_scan[:,0], x_scan[:,1], color="red", marker="X", linestyle="none")
            plt.show()

        ### CONSTANTS ###
        k = len(x_scan)//2 #100
        print(x_scan.shape)
        print(k)
        pct = 0.25 #0.25
        pct_safe = 0.0 #0.33
        
        (x_bd_unsafe, u_bd_unsafe), (x_scan_safe, u_scan_safe), counts_scan = kd_tree_detection((x_scan, u_scan), k, pct=pct)
        x_scan_unsafe = np.vstack((x_scan_unsafe, x_bd_unsafe))
        u_scan_unsafe = np.vstack((u_scan_unsafe, u_bd_unsafe))

        k_safe = len(x_scan_safe)//2 # 100
        print(x_scan_safe.shape)
        print(k_safe)

        (x_scan_filler, u_scan_filler), (x_scan_constraint, u_scan_constraint), counts_scan_safe = kd_tree_detection((x_scan_safe, u_scan_safe), k_safe, pct=pct_safe)
        x_scan_unsafe, u_scan_unsafe = mesh_controls(x_scan_unsafe, region='boundary', origin=self.pos)
        x_scan_filler, u_scan_filler = mesh_controls(x_scan_filler, region='constraint', origin=self.pos)  
        self.scan_unsafe, self.scan_filler, self.scan_constraint = (x_scan_unsafe, u_scan_unsafe), (x_scan_filler, u_scan_filler), (x_scan_constraint, u_scan_constraint)
        if plot:
            plt.figure(figsize=(6,6))
            plt.plot(x_scan_unsafe[:,0], x_scan_unsafe[:,1], color="red", marker=".", linestyle="none")
            plt.plot(x_scan_constraint[:,0], x_scan_constraint[:,1], color="c", marker=".", linestyle="none")
            #plt.plot(x_scan_filler[:,0], x_scan_filler[:,1], color="blue", marker=".", linestyle="none")
            plt.show()

        print("new position is", self.pos)
        print("new time is", self.t)
        return

    def train(self, verbose=None):  
        if verbose is not None:
            verbose_local_scope = verbose
        else:
            verbose_local_scope = self.verbose
        scan_data = (self.scan_constraint[0], self.scan_constraint[1], self.scan_filler[0], self.scan_filler[1], self.scan_unsafe[0], self.scan_unsafe[1])
        params, bias_param = solve_model(scan_data, 
        gamma_xu_constraints=GAMMA*np.ones((self.scan_constraint[0].shape[0],)),
        gamma_xu_fillers=GAMMA*np.ones((self.scan_filler[0].shape[0],)),
        safe_values=SAFE_VALUE*np.ones((self.scan_constraint[0].shape[0],)),
        unsafe_values=UNSAFE_VALUE*np.ones((self.scan_unsafe[0].shape[0],)),
        verbose=verbose_local_scope, use_casadi=True, norm_trick=True)
        self.params.append(params)
        self.bias_param.append(bias_param)
        plot_cbf(self.h, self.params, self.bias_param)
        return

    def get_waypoint(self, n_rays=5, eps=1e-3, scale=2): 

        ### Curvature ###

        theta = 2 * np.pi * np.random.rand(n_rays)
        direc = jnp.array([jnp.cos(theta), jnp.sin(theta)]).T
        zeros = []
        for d in direc:
            # 0 or self.pos?
            x_zero = binary_search(d, jnp.array([0,0]), self.h, self.params, self.bias_param, eps, scale)
            zeros.append(x_zero)
        zeros = np.array(zeros)
        hessian = jax.jacfwd(jax.jacrev(self.h))
        #curv = jnp.zeros_like(zeros)
        curv = []
        obs_ratio = jnp.zeros_like(zeros)
        for i, x in enumerate(zeros):
            # obstruction check 
            #unobs = 0
            #obs = 0
            #for d in self.data:
            #    for x_i, u_i in zip(*d):
            #        if (jnp.linalg.norm(x - x_i)   <= self.sensor_range) and 
            #           (self.seen_dict[tuple(x_i)] == 1)                 and 
            #           ( self.obs_dict[tuple(x_i)] == 1):
            #            obs += 1
            #        elif (jnp.linalg.norm(x - x_i) <= self.sensor_range):
            #            unobs += 1
            #obs_ratio[i] = obs / (obs + unobs)
            #curvature along perindicular to gradient?
            curv_x = jnp.abs(jnp.linalg.det(hessian(x, self.params, self.bias_param, RF_WEIGHTS)))
            curv.append(curv_x)
        #min_obs = jnp.min(obs_ratio)
        #max_curv = jnp.max(curv)
        #for x in zeros:
        #    way
        waypoint = zeros[np.argmax(curv)]
        return waypoint
    
    def traverse(self, t_end=10, mpc=False, info_mpc=False, track_control=False, steps=10, waypt=None, horizon=5):
        print("position is", self.pos)
        print("start time is", self.t)
        assert(((mpc==True or info_mpc==True) and track_control==False) or (mpc==False))
        if info_mpc:
            assert waypt is not None
            learned_controller = get_safe_controller(0.90, 1, self.h, self.params, self.bias_param, track_control=False,
                                                     endpt_mpc=True, pos=self.pos, x_goal=waypt, t_end=t_end, steps=steps, horizon=horizon)
        elif mpc:
            learned_controller = get_safe_controller(0.90, 1, self.h, self.params, self.bias_param, track_control=False,
                                                     mpc=True, pos=self.pos, steps=steps, t_end=t_end, horizon=horizon)
        else:
            learned_controller = get_safe_controller(0.90, 1, self.h, self.params, self.bias_param, track_control=track_control)
        learned_closed_loop = lambda t, y: dynamics(y, learned_controller(y, t))
        x0 = self.pos
        t_end = t_end        
        t_eval = np.linspace(0, t_end, num=100)
        res_learned = scipy.integrate.solve_ivp(learned_closed_loop, (0, t_end), x0, t_eval=t_eval)
        traj_learned = res_learned.y.T
        plot_cbf(self.h, self.params, self.bias_param, traj=traj_learned)
        self.pos = traj_learned[-1, :]
        self.t += t_end
        print("new position is", self.pos)
        print("new time is", self.t)
        return

 
