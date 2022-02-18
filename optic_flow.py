import numpy as np
from PIL import Image
from skimage.filters import gaussian as gaussian_filter

from image_op import *
from time import time
from multigrid import full_multigrid

class Solver():
    def __init__(self,
                 num_cycles = 3, 
                 depth_per_cycle = 3):
        self.num_cycles = num_cycles
        self.max_depths = [depth_per_cycle] * self.num_cycles
        self.base_solver = None
        
 
    def forward_diff(self, f, hx = 1.0, hy = 1.0, bc = 0):   
        curr_x = curr_y = f
            
        # Shift back/above by 1 to get f_{i+1, j}/f_{i, j+1}
        next_x = np.roll(f, -1, axis = 1)
        next_y = np.roll(f, -1, axis = 0)
        
        if bc in [0, 'neumann']:
            # Reflecting boundary conditions
            next_x[:, -1] =  next_x[:, -2]
            next_y[-1, :] =  next_y[-2, :]
        elif bc in [1, 'dirichlet']:
            # Dirichlet boundary conditions
            next_x[:, -1] =  0
            next_y[-1, :] =  0

        fx = (next_x - curr_x) / hx
        fy = (next_y - curr_y) / hy
        return fx, fy
    
    def central_diff(self, f, hx = 1.0, hy = 1.0, bc = 0):        
        # Shift back/above by 1 to get f_{i+1, j}/f_{i, j+1}
        next_x = np.roll(f, -1, axis = 1)
        next_y = np.roll(f, -1, axis = 0)
        
        # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
        prev_x = np.roll(f, 1, axis = 1)
        prev_y = np.roll(f, 1, axis = 0)
        
        # Reflecting boundary conditions
        if bc in [0, 'neumann']:
            # Reflecting boundary conditions
            next_x[:, -1] =  next_x[:, -2]
            next_y[-1, :] =  next_y[-2, :]
            prev_x[:, 0] =  prev_x[:, 1]
            prev_y[0, :] =  prev_y[1, :]
        elif bc in [1, 'dirichlet']:
            # Dirichlet boundary conditions
            next_x[:, -1] =  0
            next_y[-1, :] =  0
            prev_x[:, 0] =  0
            prev_y[0, :] =  0
        
        fx = (next_x - prev_x) / (2 * hx)
        fy = (next_y - prev_y) / (2 * hy)
        return fx, fy
        
    def backward_diff(self, f, hx = 1.0, hy = 1.0, bc = 0):
        curr_x = curr_y = f
        
        # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
        prev_x = np.roll(f, 1, axis = 1)
        prev_y = np.roll(f, 1, axis = 0)
        
        if bc in [0, 'neumann']:
            # Reflecting boundary conditions
            prev_x[:, 0] =  prev_x[:, 1]
            prev_y[0, :] =  prev_y[1, :]
        elif bc in [1, 'dirichlet']:
            # Dirichlet boundary conditions
            prev_x[:, 0] =  0
            prev_y[0, :] =  0
        
        fx = (curr_x - prev_x) / hx
        fy = (curr_y - prev_y) / hy
        return fx, fy

    
    def compute_struct_tensor(self, f1, f2, hx = 1.0, hy = 1.0, ht = 1.0, rho = 5, sigma = 0.5):
        # Compute structure tensor with smoothing in integration scale
        # rho >= 2 * sigma
        
        h, w = f1.shape[:2]
            
        f1 = f1.reshape(h, w, -1)                                       # shape = (h, w, ch)
        f2 = f2.reshape(h, w, -1)                                       # shape = (h, w, ch)
        
        f1 = gaussian_filter(f1, sigma = sigma, multichannel = True)
        f2 = gaussian_filter(f2, sigma = sigma, multichannel = True)
        
        fx1, fy1 = self.central_diff(f1, hx, hy) 
        fx2, fy2 = self.central_diff(f2, hx, hy)
        fx = (fx1 + fx2) / 2.0
        fy = (fy1 + fy2) / 2.0
        ft = (f2 - f1) / ht
        
        J11 = np.sum(gaussian_filter(fx * fx, sigma = rho, multichannel = True), axis = -1)
        J12 = np.sum(gaussian_filter(fx * fy, sigma = rho, multichannel = True), axis = -1)
        J13 = np.sum(gaussian_filter(fx * ft, sigma = rho, multichannel = True), axis = -1)
        J22 = np.sum(gaussian_filter(fy * fy, sigma = rho, multichannel = True), axis = -1)
        J23 = np.sum(gaussian_filter(fy * ft, sigma = rho, multichannel = True), axis = -1)
        J33 = np.sum(gaussian_filter(ft * ft, sigma = rho, multichannel = True), axis = -1)

        return (J11, J12, J13, J22, J23)
    
    def get_diffusivities(self, u, choice = 'charbonnier', lambd = 3., hx = 1., hy = 1.):
        # Isotropic non-linear diffiusivities
        dim = len(u.shape)
        ux, uy = self.central_diff(u, hx, hy)
        grad_sq = ux ** 2 + uy ** 2
        
        if dim == 3:
            # Couple the diffusivity computation across channels
            grad_sq = np.sum(grad_sq, axis = -1)
            
        if lambd == 0:
            return np.ones_like(grad_sq)
      
        ratio = (grad_sq) / (lambd ** 2)
        if choice in ['charbonnier', 0]:
            g = 1 / np.sqrt(1 + ratio)
        elif choice in ['perona-malik', 1]:
            g = 1 / (1 + ratio)
        elif choice in ['perona-malik', 2]:
            g = np.exp(-0.5 * ratio)
        else:
            g1 = 1
            g2 = 1. - np.exp(-3.31488 / (ratio ** 4))
            g = np.where(grad_sq == 0, g1, g2)
            
        return g
        
    def explicit_solver(self, fx, fy, ft, u = None, v = None, alpha = 500., lambd = 1., tau = 0.2, hx = 1.0, hy = 1.0):
        """ Explicit solver iteration
            Args:
        ----------------
                u: Flow vector in horizontal direction at previous iteration
                v: Flow vector in vertical direction at previous iteration
                alpha: Smoothness weight
                lambd: Lambda value in Charbonnier smoothness term
                tau: Time step size
                hx: Pixel size in x direction
                hy: Pixel size in x direction
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u: Flow vector in horizontal direction at current iteration
                v: Flow vector in vertical direction at current iteration
        """
        # Initialization
        r, c = fx.shape[:2]
        if u is None or v is None:
            u = np.zeros((r, c))
            v = np.zeros((r, c))
        
        ux1, uy1 = forward_diff(u, hx, hy)
        ux2, uy2 = backward_diff(u, hx, hy)            
        vx1, vy1 = forward_diff(v, hx, hy)
        vx2, vy2 = backward_diff(v, hx, hy)
        
        if lambd is None:
            # Create homogeneous smoothness across pixels  
            g = np.ones_like(u)
            print('here')
        else:
            uv = np.stack([u, v], axis = -1)
            g = self.get_diffusivities(uv, choice = 'charbonnier', lambd = lambd, hx = hx, hy = hy)

        # Shift back/above by 1 to get g_{i+1, j}/g_{i, j+1}
        next_gx = np.roll(g, -1, axis = 1)
        next_gy = np.roll(g, -1, axis = 0)
        
        # Shift forward/down by 1 to get g_{i-1, j}/g_{i, j-1}
        prev_gx = np.roll(g, 1, axis = 1)
        prev_gy = np.roll(g, 1, axis = 0)
        
        # Reflecting boundary conditions
        next_gx[:, -1] =  next_gy[:, -2]
        next_gy[-1, :] =  next_gx[-2, :]
        prev_gx[:, 0] =  prev_gy[:, 1]
        prev_gy[0, :] =  prev_gx[1, :]
        
        half_gx = (next_gx + g) / 2.
        neg_half_gx = (prev_gx + g) / 2.
        half_gy = (next_gy + g) / 2.
        neg_half_gy = (prev_gy + g) / 2.
        factor = tau / alpha
        u1 = (u + tau * ((half_gx * ux1 - neg_half_gx * ux2) \
                         + (half_gy * uy1 - neg_half_gy * uy2)) \
                - factor * (fx * (fy * v + ft))) / (1 + factor * fx * fx)
               
        v1 = (v + tau * ((half_gx * vx1 - neg_half_gy * vx2) \
                         + (half_gy * vy1 - neg_half_gy * vy2)) \
                - factor * (fy * (fx * u + ft))) / (1 + factor * fy * fy)
        
        return u1, v1
        
    def get_residual(self, u, v, J, alpha, h):
        # Computes residual: 
        # r^h = f^h - A^h x_tilde^h
        # This residual computation is for Laplacian in smoothness term E-L
        
        nrows, ncols = u.shape[:2]
        (J11, J12, J13, J22, J23) = J
        laplace_kernel = np.array([[0, 1, 0],
                                   [1,-4, 1],
                                   [0, 1, 0]])
        factor = (h * h) / alpha 
                              
        f_true = factor * (J12 * v + J13)
        f_est = conv2d(u, laplace_kernel) + factor * J11 * u
        
        res_u = f_true - f_est
        
        f_true = factor * (J12 * u + J23)
        f_est = conv2d(v, laplace_kernel) + factor * J22 * v
        
        res_v = f_true - f_est
        
        return res_u, res_v
    
    
    
    def cycle(self, u, v, J, depth, max_depth, alpha = 500, hx = 1, hy = 1, smoothiter = 2):
        # If scale is coarsest, return time-marching solver solution
        if depth == max_depth:
            # Use time marching solution
            u1, v1 = self.base_solver(u, v, J, alpha, hx, hy, iterations = smoothiter, parabolic = True)
            return u1, v1
            
        # Presmoothing
        t1 = time()
        u1, v1 = self.base_solver(u, v, J, alpha, hx, hy, iterations = smoothiter)
        t2 = time()
        t_diff = t2 - t1
        print('Pre-smoothing computation: {:.4f}'.format(t_diff))
                      
        (J11, J12, J13, J22, J23) = J
        
        t1 = time()
        # Compute residual
        assert hx == hy, 'Uneven grid size'
        # get new f_tilde and g_tilde
        r_u, r_v = self.get_residual(u1, v1, J, alpha, hx)
        t2 = time()
        t_diff = t2 - t1
        # print('Residual computation: {:.4f}'.format(t_diff))
        
        # Constant interpolation to keep diffusion tensor and motion tensor positive semidefinite
        # Downsample
        step = 2
        r_u_down = downsample2d(r_u, rate = step)
        r_v_down = downsample2d(r_v, rate = step)
        J11_down = downsample2d(J11, rate = step)
        J12_down = downsample2d(J12, rate = step)
        J13_down = r_u_down
        J22_down = downsample2d(J22, rate = step)
        J23_down = r_v_down
        J_down = (J11_down, J12_down, J13_down, J22_down, J23_down)
        
        # Compute errors
        e1, e2 = self.cycle(r_u_down, r_v_down, J_down, depth + 1, max_depth, alpha, step * hx, step * hy, smoothiter)
        # base_solver(r_u, r_v, J_down, alpha, hx, hy)
        
        # Upsample
        e1 = upsample2d(e1, rate = step)
        e2 = upsample2d(e2, rate = step)
        
        # Update flow vectors
        u1 += e1
        v1 += e2
        
        t2 = time()
        t_diff = t2 - t1
        print('Scale {0} computation: {1:.4f}'.format(step, t_diff))
            
        # Post-smoothing
        t1 = time()
        u1, v1 = self.base_solver(u1, v1, J, alpha, hx, hy, iterations = smoothiter)
        t2 = time()
        t_diff = t2 - t1
        print('Post-smoothing computation: {:.4f}'.format(t_diff))
        
        return u1, v1
    
    def multi_grid_solver(self, f1, f2, alpha = 500, hx = 1, hy = 1):
        """ Multi-grid solver
            Args:
        ----------------
                f1: First frame
                f2: Second frame
                alpha: Smoothness paramter
                grid_steps: Multigrid steps
            Returns:
        ----------------
                Returns the computed flow vectors:
                u: Flow vector in horizontal direction
                v: Flow vector in vertical direction
        """ 
        compute_time = []
        
        
        t1 = time()
        J = self.compute_struct_tensor(f1, f2, rho = 2.5, sigma = 0.5)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        print('Structure tensor computation: {:.4f}'.format(t_diff))
        
        # Initialization
        height, width = f1.shape[:2]
        u = np.zeros((height, width))
        v = np.zeros((height, width))
        
        for idx in range(self.num_cycles):
            # Solve using f_tilde instead of f in a cycle, for accuracy use multiple correcting multigrid cycles
            # f_tilde, g_tilde = self.compute_f_tilde(f, v)
            
            t1 = time()
            u1, v1 = self.cycle(u, v, J, depth = 1, max_depth = self.max_depths[idx], alpha = alpha, hx = 1, hy = 1, smoothiter = 2)
            compute_time.append(time() - t1)
                 
        print('Total computation time: {:.4f}'.format(sum(compute_time)))
        return (u1, v1)
        
    def jacobi(self, u, v, J, alpha = 500, hx = 1, hy = 1, iterations = 2, tau = 0.2, parabolic = False):
        """ Jacobi Solver for Non-linear system
            Args:
        ----------------
                u: Previous flow field components in horizontal direction
                v: Previous flow field components in vertical direction
                J: Motion tensor
                alpha: Smoothness paramteer
                hx, hy: Grid size
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u1: Flow fields in horizontal direction
                v1: Flow fields in vertical direction
        """
        assert hx == hy, 'Uneven grid size'
        h = hx
        (J11, J12, J13, J22, J23) = J

        nb = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
        nb_size = 4 
        factor = (h * h) / alpha  
            
        for _ in range(iterations):
            sum_nb_u = conv2d(u, nb)
            sum_nb_v = conv2d(v, nb)
            
            numr_u = sum_nb_u - factor * (J12 * v + J13)
            denr_u = nb_size + factor * J11
            numr_v  = sum_nb_v - factor * (J12 * u + J23)
            denr_v = nb_size + factor * J22
            
            if parabolic:
                numr_u = u + tau * numr_u
                denr_u = tau * denr_u + 1
                numr_v = v + tau * numr_v
                denr_v = tau * denr_v + 1
                
            u = numr_u / denr_u
            v = numr_v / denr_v
        

        return u, v

        
    def gauss_seidel(self, u, v, J, alpha = 500, hx = 1, hy = 1, iterations = 2, tau = 0.2, parabolic = False):
        """ Gauss-Seider Solver for Non-linear system
            Args:
        ----------------
                u: Previous flow field components in horizontal direction
                v: Previous flow field components in vertical direction
                J: Motion tensor
                alpha: Smoothness parameter
                hx, hy: Grid size
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u1: Flow fields in horizontal direction
                v1: Flow fields in vertical direction
        """
        # Check if causes problem on downsampling
        assert hx == hy, 'Uneven grid size'
        h = hx 
        (J11, J12, J13, J22, J23) = J
        
        nrows, ncols = u.shape[:2]
        small_nb = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 0]], dtype = bool)
        big_nb = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]], dtype = bool)
        
        
        # Can use convolution only with Jacobi
        # since for Gauss-Seidel computations depend on current timestep results
        
        # sum_small_u = self.conv2d(u1, small_nb)
        # sum_big_u = self.conv2d(u, big_nb)
        # sum_small_v = self.conv2d(v1, small_nb)
        # sum_big_v = self.conv2d(v, big_nb)
        # numr_u = sum_small_u + sum_big_u - factor * (J12 * v + J13)
        # denr_u = nb_size + factor * J11
        # numr_v  = sum_small_v + sum_big_u - factor * (J12 * v + J23)
        # denr_v = nb_size + factor * J22
        # u1 = numr_u / denr_u
        # v1 = numr_v / denr_v
        
        # TODO: Replace smoothness term (change discretization for div of diffusivity)
        # now it is simple laplacian due to norm of grad square
        
        # Neighbourhood size (in Laplacian approximation)
        nb_size = 4 
        factor = (h ** 2) / alpha
            
        omega = 1     # omega \in (0, 2) # for omega = 1, usual gs
        
        for _ in range(iterations):     
            # Reflecting bc
            u1 = np.copy(u)       
            v1 = np.copy(v)
            u1_prime = np.zeros_like(u)       
            v1_prime = np.zeros_like(u)
            u1 = np.pad(u1, 1, mode = 'symmetric')
            v1 = np.pad(v1, 1, mode = 'symmetric')
            u = np.pad(u, 1, mode = 'symmetric')
            v = np.pad(v, 1, mode = 'symmetric')     
                 
            for i in range(1, nrows):
                for j in range(1, ncols):
                    sum_small_u = np.sum((u1[i-1:i+2, j-1:j+2])[small_nb])
                    sum_small_v = np.sum((v1[i-1:i+2, j-1:j+2])[small_nb])
                    sum_big_u =  np.sum((u[i-1:i+2, j-1:j+2])[big_nb])
                    sum_big_v =  np.sum((v[i-1:i+2, j-1:j+2])[big_nb])
                    
                    numr_u = sum_small_u + sum_big_u - factor * (J12[i, j] * v[i, j] + J13[i, j])
                    denr_u = nb_size + factor * J11[i, j]
                    numr_v  = sum_small_v + sum_big_u - factor * (J12[i, j] * u[i, j] + J23[i, j])
                    denr_v = nb_size + factor * J22[i, j]
                    
                    if parabolic:
                        numr_u = u[i, j] + tau * numr_u
                        denr_u = tau * denr_u + 1
                        numr_v = v[i, j] + tau * numr_v
                        denr_v = tau * denr_v + 1
                
                    
                    # SOR: Need to tune omega with line search
                    # May be use omega matrix, since blocks of evolving structures have interacting omegas
                    u1_prime[i, j] = numr_u / denr_u
                    v1_prime[i, j] = numr_v / denr_v
                    u1[i, j] = u1[i, j] + omega * (u1_prime[i, j] - u1[i, j])
                    v1[i, j] = v1[i, j] + omega * (v1_prime[i, j] - v1[i, j])
                    
            # Remove dummy boundaries
            u = u1[1:-1, 1:-1]
            v = v1[1:-1, 1:-1]
            
        return u, v
        
    


class OpticFlow(object):
    def __init__(self, 
                 num_cycles = 10, 
                 depth_per_cycle = 3):
        self.solver = Solver(num_cycles = num_cycles, depth_per_cycle = depth_per_cycle)
        
    def __call__(self):
        return self.compute_flow()

    def visualize(self, u, v):
        """ Computes RGB image visualizing the flow vectors """
        max_mag = np.amax(np.sqrt(u ** 2 + v ** 2))

        u = u / max_mag
        v = v / max_mag
        
        angle = np.where(u == 0., 0.5 * np.pi, np.arctan(v / u))
        angle[(u == 0) * (v < 0.)] += np.pi
        angle[u < 0.] += np.pi
        angle[(u > 0.) * (v < 0.)] += 2 * np.pi
        
        r = np.zeros_like(u, dtype = float)
        g = np.zeros_like(u, dtype = float)
        b = np.zeros_like(u, dtype = float)
        
        mag = np.minimum(np.sqrt(u ** 2 + v ** 2), 1.)
        
        # Red-Blue Case
        case = (angle >= 0.0) * (angle < 0.25 * np.pi)
        a = angle / (0.25 * np.pi)
        r = np.where(case, a * 255. + (1 - a) * 255., r)
        b = np.where(case, a * 255. + (1 - a) * 0., b)
        case = (angle >= 0.25 * np.pi) * (angle < 0.5 * np.pi)
        a = (angle - 0.25 * np.pi) / (0.25 * np.pi)
        r = np.where(case, a * 64. + (1 - a) * 255., r)
        g = np.where(case, a * 64. + (1 - a) * 0., g)
        b = np.where(case, a * 255. + (1 - a) * 255., b)
        
        # Blue-Green Case
        case = (angle >= 0.5 * np.pi) * (angle < 0.75 * np.pi)
        a = (angle - 0.5 * np.pi) / (0.25 * np.pi)
        r = np.where(case, a * 0. + (1 - a) * 64., r)
        g = np.where(case, a * 255. + (1 - a) * 64., g)
        b = np.where(case, a * 255. + (1 - a) * 255., b)
        case = (angle >= 0.75 * np.pi) * (angle < np.pi)
        a = (angle - 0.75 * np.pi) / (0.25 * np.pi)
        g = np.where(case, a * 255. + (1 - a) * 255., g)
        b = np.where(case, a * 0. + (1 - a) * 255., b)
        
        # Green-Yellow Case
        case = (angle >= np.pi) * (angle < 1.5 * np.pi)
        a = (angle - np.pi) / (0.5 * np.pi)
        r = np.where(case, a * 255. + (1 - a) * 0., r)
        g = np.where(case, a * 255. + (1 - a) * 255., g)
        
        # Yellow-Red Case        
        case = (angle >= 1.5 * np.pi) * (angle < 2. * np.pi)
        a = (angle - 1.5 * np.pi) / (0.5 * np.pi)
        r = np.where(case, a * 255. + (1 - a) * 255., r)
        g = np.where(case, a * 0. + (1 - a) * 255., g)
        
        r = np.minimum(np.maximum(r * mag, 0.0), 255.)
        g = np.minimum(np.maximum(g * mag, 0.0), 255.)
        b = np.minimum(np.maximum(b * mag, 0.0), 255.)
        
        flow_img = np.stack([r, g, b], axis = -1).astype(np.uint8)
        
        return flow_img
        
    def compute_flow(self, f1, f2, alpha = 500, lambd = 1, tau = 0.2, maxiter = 1000, solver = 'explicit', base_solver = 'jacobi'):
        r, c = f1.shape[:2]
        
        u = v = np.zeros((r, c))
        
        if base_solver == 'jacobi':
            self.solver.base_solver = self.solver.jacobi
        else:
            self.solver.base_solver = self.solver.gauss_seidel
            
        print('Alpha:', alpha)
        print('Lambda:', lambd)
        print('tau:', tau)
        
        if solver == 'explicit':
            fx, fy, ft = get_derivatives(f1, f2)
            for it in range(maxiter):
                u, v = self.solver.explicit_solver(fx, fy, ft, u, v, alpha = alpha, lambd = lambd)
                mag = np.sqrt(u ** 2 + v ** 2)
                print('{}: Max mag: {:.2f} Mean mag: {:.2f}'.format(it, np.amax(mag), np.mean(mag)))
        else:
            u, v = self.solver.multi_grid_solver(f1, f2, alpha = alpha)
        
        
        vis = self.visualize(u, v)
        return vis
        
    
def main():
    kmax = 200              # Number of iterations (we are interested in steady state of the diffusion-reaction system)
    alpha = 500             # Regularization Parameter (should be large enough to weight smoothness terms which have small magnitude)
    tau = 0.2               # Step size (For implicit scheme, can choose arbitrarily large, for explicit scheme  <=0.25)
    lambd = 0.1
    solver = 'multi_grid'
    base_solver = 'jacobi'
    # frame1_path = input('Enter first image: ')
    # frame2_path = input('Enter second image: ')
    frame1_path = 'a.pgm'
    frame2_path = 'b.pgm'
    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    
    f1 = np.array(frame1, dtype = np.float)
    f2 = np.array(frame2, dtype = np.float)
    
    optic_flow = OpticFlow(num_cycles = 3, depth_per_cycle = 3)
    vis = optic_flow.compute_flow(f1, f2, alpha = alpha, lambd = lambd, tau = tau, maxiter = kmax, solver = solver, base_solver = base_solver)
    vis = Image.fromarray(vis)
    vis.save('./visual.pgm')
    vis.show()

if __name__ == '__main__':
    main()    
    
