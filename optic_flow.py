import numpy as np
from PIL import Image
from skimage.filters import gaussian as gaussian_filter

from image_op import *
from time import time
from multigrid import full_multigrid

class OpticFlow():
    def __init__(self,
                 num_cycles = 3, 
                 depth_per_cycle = 3,
                 smooth_iter = 2,
                 noise_scale = 0.5,
                 integration_scale = 5,
                 hx = 1,
                 hy = 1,
                 ht = 1,
                 tau = 0.2,
                 lambd = 5,
                 alpha = 500,
                 base_solver = 'jacobi'):
        self.num_cycles = num_cycles
        self.max_depths = [depth_per_cycle] * self.num_cycles
        
        self.smooth_iter = smooth_iter
        self.noise_scale = noise_scale
        self.integration_scale = integration_scale
        
        self.hx = hx
        self.hy = hy
        self.ht = ht
        
        self.tau = tau
        self.alpha = alpha
        self.lambd = lambd
        
        if base_solver == 'gs':
            self.base_solver = self.gauss_seidel
        else:
            self.base_solver = self.jacobi
        
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

    
    def compute_struct_tensor(self, f1, f2):
        # Compute structure tensor with smoothing in integration scale
        # rho >= 2 * sigma
        
        h, w = f1.shape[:2]
            
        f1 = f1.reshape(h, w, -1)                                       # shape = (h, w, ch)
        f2 = f2.reshape(h, w, -1)                                       # shape = (h, w, ch)
        
        f1 = gaussian_filter(f1, sigma = self.noise_scale, multichannel = True)
        f2 = gaussian_filter(f2, sigma = self.noise_scale, multichannel = True)
        
        fx1, fy1 = self.central_diff(f1, self.hx, self.hy) 
        fx2, fy2 = self.central_diff(f2, self.hx, self.hy)
        fx = (fx1 + fx2) / 2.0
        fy = (fy1 + fy2) / 2.0
        ft = (f2 - f1) / self.ht
        
        J11 = np.sum(gaussian_filter(fx * fx, sigma = self.integration_scale, multichannel = True), axis = -1)
        J12 = np.sum(gaussian_filter(fx * fy, sigma = self.integration_scale, multichannel = True), axis = -1)
        J13 = np.sum(gaussian_filter(fx * ft, sigma = self.integration_scale, multichannel = True), axis = -1)
        J22 = np.sum(gaussian_filter(fy * fy, sigma = self.integration_scale, multichannel = True), axis = -1)
        J23 = np.sum(gaussian_filter(fy * ft, sigma = self.integration_scale, multichannel = True), axis = -1)
        J33 = np.sum(gaussian_filter(ft * ft, sigma = self.integration_scale, multichannel = True), axis = -1)
        
        nabla_fx = np.sum(fx * fx, axis = -1) ** 0.5
        nabla_fy = np.sum(fy * fy, axis = -1) ** 0.5
        nabla_f = np.stack([nabla_fx, nabla_fy], axis = -1)
        
        g = self.get_diffusivities(nabla_f)
        print(g.shape)

        return (J11, J12, J13, J22, J23), g
    
    def get_diffusivities(self, nabla_f, choice = 'charbonnier', lambd = 3., hx = 1., hy = 1.):
        # Isotropic non-linear diffiusivities
        dim = len(nabla_f.shape)
        #ux, uy = self.central_diff(u, hx, hy)
        #grad_sq = ux ** 2 + uy ** 2
        
        grad_sq = nabla_f * nabla_f
        print(grad_sq.shape)
        if dim == 3:
            # Couple the diffusivity computation across channels
            grad_sq = np.sum(grad_sq, axis = -1)
        print(grad_sq.shape)
            
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
       
    def get_residual(self, u, v, J, g, hx, hy):
        # Computes residual: 
        # r^h = f^h - A^h x_tilde^h
        # This residual computation is for Laplacian in smoothness term E-L
        
        nrows, ncols = u.shape[:2]
        (J11, J12, J13, J22, J23) = J
        laplace_kernel = np.array([[0, 1, 0],
                                   [1,-4, 1],
                                   [0, 1, 0]])
        factor = 1 / self.alpha 
        
        b_1 = factor * (J12 * v + J13)
        
        sum_nb_u, sum_nb_v, center_wt = self.compute_diffusion_term(u, v, g, hx, hy)
        A_u = sum_nb_u - center_wt * u + factor * J11 * u
        
        res_u = b_1 - A_u
        
        b_2 = factor * (J12 * u + J23)
        A_v = sum_nb_v - center_wt * v + factor * J22 * v
        
        res_v = b_2 - A_v
        
        #factor = (h * h) / self.alpha                       
        #f_true = factor * (J12 * v + J13)
        #f_est = conv2d(u, laplace_kernel) + factor * J11 * u
        #res_u = f_true - f_est
        
        #f_true = factor * (J12 * u + J23)
        #f_est = conv2d(v, laplace_kernel) + factor * J22 * v
        #res_v = f_true - f_est
        
        return res_u, res_v
    
    
    
    def cycle(self, u, v, J, g, depth, max_depth, hx = 1, hy = 1):
        # If scale is coarsest, return time-marching solver solution
        if depth == max_depth:
            # Use time marching solution
            u1, v1 = self.base_solver(u, v, J, g, hx, hy, iterations = self.smooth_iter, parabolic = True)
            return u1, v1
            
        # Presmoothing
        t1 = time()
        u1, v1 = self.base_solver(u, v, J, g, hx, hy, iterations = self.smooth_iter)
        t2 = time()
        t_diff = t2 - t1
        # print('Pre-smoothing computation: {:.4f}'.format(t_diff))
                      
        (J11, J12, J13, J22, J23) = J
        
        t1 = time()
        # Compute residual
        assert hx == hy, 'Uneven grid size'
        # get new f_tilde and g_tilde
        r_u, r_v = self.get_residual(u1, v1, J, g, hx, hy)
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
        g_down = downsample2d(g, rate = step)
        
        # Compute errors
        e1, e2 = self.cycle(np.zeros_like(r_u_down), np.zeros_like(r_v_down), J_down, g_down, depth + 1, max_depth, step * hx, step * hy)
        # base_solver(r_u, r_v, J_down, alpha, hx, hy)
        
        # Upsample
        e1 = upsample2d(e1, rate = step)
        e2 = upsample2d(e2, rate = step)
        
        # Update flow vectors
        u1 += e1
        v1 += e2
        
        t2 = time()
        t_diff = t2 - t1
        # print('Scale {0} computation: {1:.4f}'.format(step, t_diff))
            
        # Post-smoothing
        t1 = time()
        u1, v1 = self.base_solver(u1, v1, J, g, hx, hy, iterations = self.smooth_iter)
        t2 = time()
        t_diff = t2 - t1
        # print('Post-smoothing computation: {:.4f}'.format(t_diff))
        
        return u1, v1
    
    def multi_grid_solver(self, f1, f2):
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
        J, g = self.compute_struct_tensor(f1, f2)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        # print('Structure tensor computation: {:.4f}'.format(t_diff))
        
        # Initialization
        height, width = f1.shape[:2]
        u = np.zeros((height, width))
        v = np.zeros((height, width))
        
        for idx in range(self.num_cycles):
            # Solve using f_tilde instead of f in a cycle, for accuracy use multiple correcting multigrid cycles
            # f_tilde, g_tilde = self.compute_f_tilde(f, v)
            
            t1 = time()
            u, v = self.cycle(u, v, J, g, depth = 1, max_depth = self.max_depths[idx], hx = 1, hy = 1)
            compute_time.append(time() - t1)
            mag = np.sqrt(u ** 2 + v ** 2)
            print('Cycle {}: Max mag: {:.2f} Mean mag: {:.2f}'.format(idx, np.amax(mag), np.mean(mag)))
                 
        print('Total computation time: {:.4f}'.format(sum(compute_time)))
        return (u, v)
        
    def get_diffusivities_old(self, u, choice = 'charbonnier', hx = 1., hy = 1.):
        ones = np.ones((u.shape[1], u.shape[2]))
        if self.lambd == 0:
            return ones
            
        # Isotropic non-linear diffiusivities
        dim = len(u.shape)
        ux, uy = self.central_diff(u, hx, hy)
        grad_sq = ux ** 2 + uy ** 2

        # Couple the diffusivity computation across channels
        grad_sq = np.sum(grad_sq, axis = 0)
            
      
        ratio = (grad_sq) / (self.lambd ** 2)
        if choice in ['charbonnier', 0]:
            g = 1 / np.sqrt(1 + ratio)
        elif choice in ['perona-malik', 1]:
            g = 1 / (1 + ratio)
        elif choice in ['perona-malik', 2]:
            g = np.exp(-0.5 * ratio)
        else:
            g1 = ones
            g2 = 1. - np.exp(-3.31488 / (ratio ** 4))
            g = np.where(grad_sq == 0, g1, g2)
            
        return g
    
    
    def get_shifted(self, f):
        # Shift back/above by 1 to get f_{i+1, j}/f_{i, j+1}
        next_fx = np.roll(f, -1, axis = 1)
        next_fy = np.roll(f, -1, axis = 0)
        
        # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
        prev_fx = np.roll(f, 1, axis = 1)
        prev_fy = np.roll(f, 1, axis = 0)
        
        # Reflecting boundary conditions
        next_fx[:, -1] = next_fx[:, -2]
        next_fy[-1, :] = next_fy[-2, :]
        prev_fx[:, 0] = prev_fx[:, 1]
        prev_fy[0, :] = prev_fy[1, :]
    
        return next_fx, next_fy, prev_fx, prev_fy
    
    def compute_diffusion_term(self, u, v, g, hx = 1, hy = 1, homogeneous = False):
        factor = 1.0 / (hx * hx)
        if False and (homogeneous or self.lambd == 0):
            nb = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
            sum_nb_u = factor * conv2d(u, nb)
            sum_nb_v = factor * conv2d(v, nb)
            return sum_nb_u, sum_nb_v, 4 * factor
            
        ux1, uy1 = self.forward_diff(u, hx, hy)
        ux2, uy2 = self.backward_diff(u, hx, hy)            
        vx1, vy1 = self.forward_diff(v, hx, hy)
        vx2, vy2 = self.backward_diff(v, hx, hy)
        
        # uv = np.stack([u, v], axis = 0)
        
        next_gx, next_gy, prev_gx, prev_gy = self.get_shifted(g)
        next_ux, next_uy, prev_ux, prev_uy = self.get_shifted(u)
        next_vx, next_vy, prev_vx, prev_vy = self.get_shifted(v)
        
        next_half_gx = (next_gx + g) / 2.
        prev_half_gx = (prev_gx + g) / 2.
        next_half_gy = (next_gy + g) / 2.
        prev_half_gy = (prev_gy + g) / 2.
        
        sum_nb_u = factor * ((next_half_gx * next_ux + prev_half_gx * prev_ux) \
                  + (next_half_gy * next_uy + prev_half_gy * prev_uy))
        sum_nb_v = factor * ((next_half_gx * next_vx + prev_half_gx * prev_vx) \
                  + (next_half_gy * next_vy + prev_half_gy * prev_vy))
        center_wt = factor * (next_half_gx + prev_half_gx + next_half_gy + prev_half_gy)
        
        # sum_nb_u = (1 / h) * ((next_half_gx * ux1 - prev_half_gx * ux2) \
        #          + (next_half_gy * uy1 - prev_half_gy * uy2))
        # sum_nb_v = (1 / h) * ((next_half_gx * vx1 - prev_half_gx * vx2) \
        #          + (next_half_gy * vy1 - prev_half_gy * vy2))
                  
        return sum_nb_u, sum_nb_v, center_wt
        
    def jacobi(self, u, v, J, g, hx = 1, hy = 1, iterations = 2, parabolic = False):
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
        #factor = (h * h) / self.alpha  
        factor = 1 / self.alpha  
            
        for _ in range(iterations):
            #sum_nb_u = conv2d(u, nb)
            #sum_nb_v = conv2d(v, nb)
            
            #numr_u = sum_nb_u - factor * (J12 * v + J13)
            #denr_u = nb_size + factor * J11
            #numr_v  = sum_nb_v - factor * (J12 * u + J23)
            #denr_v = nb_size + factor * J22
            
            
            sum_nb_u, sum_nb_v, center_wt = self.compute_diffusion_term(u, v, g, hx, hy)
            numr_u = sum_nb_u - factor * (J12 * v + J13)
            denr_u = center_wt + factor * J11
            numr_v  = sum_nb_v - factor * (J12 * u + J23)
            denr_v = center_wt + factor * J22
            
            if parabolic:
                numr_u = u + self.tau * numr_u
                denr_u = self.tau * denr_u + 1
                numr_v = v + self.tau * numr_v
                denr_v = self.tau * denr_v + 1
                
            u = numr_u / denr_u
            v = numr_v / denr_v
        

        return u, v

        
    def gauss_seidel(self, u, v, J, hx = 1, hy = 1, iterations = 2, parabolic = False):
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
        
        
        # Neighbourhood size (in Laplacian approximation)
        nb_size = 4 
        factor = (h ** 2) / self.alpha
            
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
                        numr_u = u[i, j] + self.tau * numr_u
                        denr_u = self.tau * denr_u + 1
                        numr_v = v[i, j] + self.tau * numr_v
                        denr_v = self.tau * denr_v + 1
                
                    
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
        
    
    def __call__(self, f1, f2):            
        print('Alpha:', self.alpha)
        print('Lambda:', self.lambd)
        print('tau:', self.tau)
        
        u, v = self.multi_grid_solver(f1, f2)
        
        vis = self.visualize(u, v)
        return vis

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
        # max_val = np.amax(flow_img)
        # flow_img = 255 * flow_img / max_val 
        # flow_img = flow_img.astype(np.uint8)
        
        return flow_img


        
    
def main():
    smooth_iter = 3         # Number of iterations (we are interested in steady state of the diffusion-reaction system)
    alpha = 100              # Regularization Parameter (should be large enough to weight smoothness terms which have small magnitude)
    tau = 0.2               # Step size (For implicit scheme, can choose arbitrarily large, for explicit scheme  <=0.25)
    lambd = 4              # Contrast parameter used in diffusivity
    solver = 'multigrid'
    base_solver = 'jacobi'
    noise_scale = 0.5
    integration_scale = 3
    
    
    # frame1_path = input('Enter first image: ')
    # frame2_path = input('Enter second image: ')
    frame1_path = 'test/1.png'
    frame2_path = 'test/2.png'
    #frame1_path = 'a.pgm'
    #frame2_path = 'b.pgm'
    frame1 = Image.open(frame1_path).convert('RGB').resize((456, 256))
    frame2 = Image.open(frame2_path).convert('RGB').resize((456, 256))
    
    f1 = np.array(frame1, dtype = np.float)
    f2 = np.array(frame2, dtype = np.float)
    
    optic_flow = OpticFlow(num_cycles = 4, 
                           depth_per_cycle = 3,
                           noise_scale = noise_scale,
                           integration_scale = integration_scale,
                           alpha = alpha, 
                           lambd = lambd, 
                           tau = tau, 
                           smooth_iter = smooth_iter,
                           base_solver = base_solver)
                          
    vis = optic_flow(f1, f2)
    vis = Image.fromarray(vis)
    vis.save('./visual.pgm')
    vis.show()

if __name__ == '__main__':
    main()    
    
