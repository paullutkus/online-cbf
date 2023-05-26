import numpy as np
from scipy.stats import rankdata

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
    inner_x, inner_u = generate_mesh((0, radius-boundary_pct*radius), density, region='inner_filler')
    boundary_x, boundary_u = generate_mesh((radius-boundary_pct*radius, radius), density, region='boundary')
    inner, boundary = (inner_x, inner_u), (boundary_x, boundary_u)
    return inner, boundary
        






