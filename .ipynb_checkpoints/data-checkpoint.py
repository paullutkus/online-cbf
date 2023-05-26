import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.neighbors import KDTree
from controls import get_feedback_linear_controller, get_xref
from dynamics import dynamics

###################
### Create mesh ###
###################

def t(r, x):
    t = np.arccos(-x[0]/r) # [0, pi]
    t_y = np.arcsin(x[1]/r) # [-pi/2, pi/2]
    if t_y < 0:
        t = (2 * np.pi) - t
    return t

def mesh_controls(x_pts, region=None, origin=(0, 0)):
    u_pts = []
    new_x_pts = []
    while len(x_pts) > 0:
        norms = np.linalg.norm(x_pts - origin, axis=1)
        norm_sort = rankdata(norms, method='min')
        selected = x_pts[norm_sort == 1]
        x_pts = x_pts[norm_sort != 1]
        for x in selected:
            r = np.linalg.norm(x - origin)
            t_x = t(r, x - origin)
            if region == 'constraint' or region == 'boundary':
                controller = get_feedback_linear_controller(get_xref(r, origin), 1)
                u = controller(x, t_x)
            elif region == 'inner_filler':
                u = (1/(x**2 + 1)) * 2 * x
            else:
                u = np.array([0,0])
            new_x_pts.append(x)
            u_pts.append(u)
    new_x_pts = np.array(new_x_pts)
    u_pts = np.array(u_pts)
    return (new_x_pts, u_pts)

def generate_mesh(boundary, density, region):
    n_pts = int(2*boundary[1] * density)
    ax = np.linspace(-boundary[1], boundary[1], num=n_pts)
    grid = np.einsum('ijk->kji', np.meshgrid(ax, ax)).reshape(ax.shape[0]**2, 2)
    def cond(grid, boundary):
        inner_cond = np.linalg.norm(grid, axis=1) > boundary[0]
        outer_cond = np.linalg.norm(grid, axis=1) <= boundary[1]
        return np.multiply(inner_cond, outer_cond)
    mesh = grid[cond(grid, boundary)]
    x_pts, u_pts = mesh_controls(mesh, region)
    return (x_pts, u_pts)

def generate_radial_mesh(radius=0.8, density=33, boundary_pct=0.25):
    inner_x, inner_u = generate_mesh((0, radius-boundary_pct*radius), density, region='outer_filler')
    boundary_x, boundary_u = generate_mesh((radius-boundary_pct*radius, radius), density, region='boundary')
    inner, boundary = (inner_x, inner_u), (boundary_x, boundary_u)
    return inner, boundary

#######################
### Generate shapes ###
#######################

def generate_rectangle(height=1, width=1, density=200, unsafe_margin=0.25, center=(0,0)):

    # Create rectangular grid
    n_pts = int( (height+max(height,width)*unsafe_margin) * (width+max(height,width)*unsafe_margin) * density )
    ax_1 = np.linspace( -(width + max(height,width) * unsafe_margin) / 2 + center[0], 
                         (width + max(height,width) * unsafe_margin) / 2 + center[0], 
                         int((width + max(height,width) * unsafe_margin) * np.sqrt(density)) )

    ax_2 = np.linspace( -(height + max(height,width) * unsafe_margin) / 2 + center[1], 
                         (height + max(height,width) * unsafe_margin) / 2 + center[1],
                         int((height + max(height,width) * unsafe_margin) * np.sqrt(density)) ) 

    grid = np.einsum('ijk->kji', np.meshgrid(ax_1, ax_2)).reshape(ax_1.shape[0] * ax_2.shape[0], 2)

    # Create obstacle information dictionary
    in_width  = lambda x : True if (x <= width  / 2 + center[0]) and (x >= -width  / 2 + center[0]) else False
    in_height = lambda y : True if (y <= height / 2 + center[1]) and (y >= -height / 2 + center[1]) else False
    obs_dict = {tuple(pt) : 1 if not (in_width(pt[0]) and in_height(pt[1])) else 0 for pt in grid}

    return grid, obs_dict  

def insert_shape(pos, grid, obs_dict, shape='triangle', scale=0.2, theta=0):
    if shape == 'triangle':
        # Create three hyperplanes, check to see whether it falls on the correct side of each 
        # Alternative condition: angles w.r.t. vertices add up to 360 inside triangle, < 360 outside

        # Vertices are v1: top, v2: bottom left, v3 : bottom right
        # There are constraints on the angle where this is valid!
        assert(abs(theta) < np.pi/2)
        v1 = np.array((pos[0] - scale * np.sin(theta)            , pos[1] + scale * np.cos(theta)))
        v2 = np.array((pos[0] - scale * np.sin(theta + 2*np.pi/3), pos[1] + scale * np.cos(theta + 2*np.pi/3)))
        v3 = np.array((pos[0] - scale * np.sin(theta + 4*np.pi/3), pos[1] + scale * np.cos(theta + 4*np.pi/3)))
        v1v3 = lambda x : (v1[1]-v3[1])/(v1[0]-v3[0]) * x + (v1[1] - (v1[1]-v3[1])/(v1[0]-v3[0])*v1[0])
        v2v3 = lambda x : (v2[1]-v3[1])/(v2[0]-v3[0]) * x + (v2[1] - (v2[1]-v3[1])/(v2[0]-v3[0])*v2[0])
        v1v2 = lambda x : (v1[1]-v2[1])/(v1[0]-v2[0]) * x + (v1[1] - (v1[1]-v3[1])/(v1[0]-v2[0])*v1[0])
        triangle_pts = [tuple((x[0], x[1])) for x in grid if ((v1v3(x[0]) >= x[1]) and (v1v2(x[0]) >= x[1]) and (v2v3(x[0]) <= x[1]))]
        for pt in triangle_pts: obs_dict[pt] = 2
    
    elif shape == 'rhombus':
        for k, x in enumerate(grid):
            x_trans = x - np.array(pos)
            x_scale = 1/scale * x_trans
            rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
            x_derotated = rot @ x_scale
            if np.linalg.norm(x_derotated, ord=np.inf) <= 1:
                obs_dict[tuple(x)] = 3

            
        '''
        grid_translated = [x - pos for x in grid]
        grid_scaled = [1/scale * x for x in grid_translated]
        rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                        [np.sin(-theta), np.cos(-theta)]])
        grid_derotated = [rot @ x for x in grid_scaled]
        rhombus_pts = [tuple(x) for k, x in enumerate(grid) if np.linalg.norm(grid_derotated[k] <= 1, ord=np.inf)]
        for pt in rhombus_pts: 
            obs_dict[pt] = 3
        '''

    elif shape == 'circle':
        for k, x in enumerate(grid):
            x_trans = x - np.array(pos)
            x_scale = 1/scale * x_trans
            if np.linalg.norm(x_scale) <= 1:
                obs_dict[tuple(x)] = 4
        '''
        grid_translated = [x - pos for x in grid]
        grid_scaled = [1/scale * x for x in grid_translated]
        circle_pts = [tuple(x) for k, x in enumerate(grid) if np.linalg.norm(grid_scaled[k] <= 1)]
        for pt in circle_pts: 
            obs_dict[pt] = 4
        '''

    return obs_dict

        
#################
### Plot data ###
#################

def plot_data(grid, obs_dict=None, u=None, plot_dynamics=False):
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(9, 9)) 

    assert(grid is not None and obs_dict is not None)   
    color_dict = {0:"yellow",  1:"red",  2:"green",  3:"red",  4:"blue"}
    for x in grid:
        color = color_dict[obs_dict[tuple(x)]]
        if plot_dynamics:
            assert(u is not None)
            fxu = dynamics(xpt, upt)
            d = 0.01 * (fxu / jnp.linalg.norm(fxu))
            plt.arrow(xpt[0], xpt[1], d[0], d[1], edgecolor=color, facecolor=color, head_width=0.01, head_length=0.01)  
        plt.plot(x[0], x[1], color=color, marker=".", linestyle="none")

    plt.grid()
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    plt.tight_layout(); return

def plot_shape_data(grid, obs_dict):
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.figure(figsize=(9, 9))
    for x in grid:
        if obs_dict[tuple(x)] == 0:
            plt.plot(x[0], x[1], color="yellow", marker=".", linestyle="none")
        if obs_dict[tuple(x)] == 1:
            plt.plot(x[0], x[1], color="red", marker=".", linestyle="none")
        if obs_dict[tuple(x)] == 2:
            plt.plot(x[0], x[1], color="green", marker=".", linestyle="none")
        if obs_dict[tuple(x)] == 3:
            plt.plot(x[0], x[1], color="red", marker=".", linestyle="none")
        if obs_dict[tuple(x)] == 4:
            plt.plot(x[0], x[1], color="blue", marker=".", linestyle="none")
    plt.show() 
    return None

################################
### Boundary point detection ###
################################

def kd_tree_detection(data, k, eta=None, pct=None):
    Z_safe, u_safe = data
    tree = KDTree(Z_safe)
    _, knn_inds = tree.query(Z_safe, k=k)
    flat_inds = knn_inds.flatten()
    counts = np.bincount(flat_inds)
    if pct is not None:
        pct_unsafe = pct
        nbr_thresh = np.quantile(counts, pct_unsafe)
        Z_N = Z_safe[counts < nbr_thresh]; u_N = u_safe[counts < nbr_thresh]
        Z_Nc = Z_safe[counts >= nbr_thresh]; u_Nc = u_safe[counts >= nbr_thresh]
    elif eta is not None:
        Z_N = Z_safe[counts < eta]; u_N = u_safe[counts < eta]
        Z_Nc = Z_safe[counts >= eta]; u_Nc = u_safe[counts >= eta]
    else:
        pct_unsafe = 0.2
        nbr_thresh = np.quantile(counts, pct_unsafe)
        Z_N = Z_safe[counts < nbr_thresh]; u_N = u_safe[counts < nbr_thresh]
        Z_Nc = Z_safe[counts >= nbr_thresh]; u_Nc = u_safe[counts >= nbr_thresh]

    data_N = (Z_N, u_N)
    data_Nc = (Z_Nc, u_Nc)
    return data_N, data_Nc, counts


