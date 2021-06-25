from image_op import *
from time import time
from multigrid import full_multigrid

class Solver():
    def __init__(self):
        pass
        
    def struct_tensor(self, fx, fy, ft, rho = 5):
        # Compute structure tensor with smoothing in integration scale
        # rho >= 2 * sigma
        J11 = fx * fx
        J12 = fx * fy
        J13 = fx * ft
        J22 = fy * fy
        J23 = fy * ft
        J33 = ft * ft
        if rho > 0:
            J11 = fft_gauss(J11, rho)
            J12 = fft_gauss(J12, rho)
            J13 = fft_gauss(J13, rho)
            J22 = fft_gauss(J22, rho)
            J23 = fft_gauss(J23, rho)
            J33 = fft_gauss(J33, rho)
        return (J11, J12, J13, J22, J23)
    
    def get_diffusivities(self, u, choice = 'charbonnier', lambd = 3., hx = 1., hy = 1.):
        # Isotropic non-linear diffiusivities
        dim = len(u.shape)
        ux, uy = central_diff(u, hx, hy)
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
        
    def explicit_solver(self, fx, fy, ft, u = None, v = None, alpha = 500., lambd = 4., tau = 0.2, hx = 1.0, hy = 1.0):
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
        nrows, ncols = u.shape[:2]
        (J11, J12, J13, J22, J23) = J
        laplace_kernel = np.array([[0, 1, 0],
                                   [1,-4, 1],
                                   [0, 1, 0]])
        factor =  (h * h / alpha)                         
        f_true = -factor * (J12 * v + J13)
        f_est = conv2d(u, laplace_kernel)
        res_u = f_true - f_est
        f_true = -factor * (J12 * u + J23)
        f_est = conv2d(v, laplace_kernel)
        res_v = f_true - f_est
        
        return res_u, res_v
    
    def multi_grid_solver(self, f1, f2, grid_steps = [2, 4, 8], alpha = 500, h = 1):
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
        # Initialization
        r, c = f1.shape[:2]
        u = np.zeros((r, c))
        v = np.zeros((r, c))
        
        fx, fy, ft = get_derivatives(f1, f2)
        
        t1 = time()
        J = self.struct_tensor(fx, fy, ft, rho = 5)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        print('Structure tensor computation: {:.4f}'.format(t_diff))
        
        # Presmoothing
        t1 = time()
        u1, v1 = self.gauss_seidel(u, v, J, alpha, h)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        print('Pre-smoothing computation: {:.4f}'.format(t_diff))
        
        # A discretization of laplacian, or div(g)
        # grid_steps = [1, 2, 4, 2, 1]       
        grid_steps = full_multigrid(cycles = 1, depth = 3)

                      
        (J11, J12, J13, J22, J23) = J
        for step in grid_steps:
            t1 = time()
            # Compute residual
            r_u, r_v = self.get_residual(u1, v1, J, alpha, h)
            t2 = time()
            t_diff = t2 - t1
            compute_time.append(t_diff)
            # print('Residual computation: {:.4f}'.format(t_diff))
            
            # Constant interpolation to keep diffusion tensor and motion tensor positive semidefinite
            # Downsample
            J11_down = downsample2d(J11, rate = step)
            J12_down = downsample2d(J12, rate = step)
            J13_down = downsample2d(J13, rate = step)
            J22_down = downsample2d(J22, rate = step)
            J23_down = downsample2d(J23, rate = step)
            r_u = downsample2d(r_u, rate = step)
            r_v = downsample2d(r_v, rate = step)
            J_down = (J11_down, J12_down, J13_down, J22_down, J23_down)
            
            # h = step 
            # Compute errors
            e1, e2 = self.gauss_seidel(r_u, r_v, J_down, h = h)
            
            # Upsample
            e1 = upsample2d(e1, rate = step)
            e2 = upsample2d(e2, rate = step)
            # Update flow vectors
            u1 += e1
            v1 += e2
            
            t2 = time()
            t_diff = t2 - t1
            compute_time.append(t_diff)
            print('Scale {0} computation: {1:.4f}'.format(step, t_diff))
            
        # Post-smoothing
        t1 = time()
        u1, v1 = self.gauss_seidel(u1, v1, J, alpha, h)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        print('Post-smoothing computation: {:.4f}'.format(t_diff))
        
        print('Total computation time: {:.4f}'.format(sum(compute_time)))
        return (u1, v1)
        
    def jacobi(self, u, v, J, alpha = 500, h = 1):
        """ Jacobi Solver for Non-linear system
            Args:
        ----------------
                u: Previous flow fields in horizontal direction
                v: Previous flow fields in vertical direction
                J: Motion tensor
                alpha: Smoothness paramter
                h: Grid size
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u1: Flow fields in horizontal direction
                v1: Flow fields in vertical direction
        """
        (J11, J12, J13, J22, J23) = J

        nb = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
        nb_size = 4 
        factor = (h ** 2) / alpha  
        
        sum_nb_u = conv2d(u, nb)
        sum_nb_v = conv2d(v, nb)
        
        numr_u = sum_nb_u - factor * (J12 * v + J13)
        denr_u = nb_size + factor * J11
        numr_v  = sum_nb_v - factor * (J12 * u + J23)
        denr_v = nb_size + factor * J22
        
        u1 = numr_u / denr_u
        v1 = numr_v / denr_v

        return u1, v1

        
    def gauss_seidel(self, u, v, J, alpha = 500, h = 1):
        """ Gauss-Seider Solver for Non-linear system
            Args:
        ----------------
                u: Previous flow fields in horizontal direction
                v: Previous flow fields in vertical direction
                J: Motion tensor
                alpha: Smoothness paramter
                h: Grid size
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u1: Flow fields in horizontal direction
                v1: Flow fields in vertical direction
        """
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
             
        # Reflecting bc
        u1 = np.copy(u)       
        v1 = np.copy(v)
        u1_prime = np.zeros_like(u)       
        v1_prime = np.zeros_like(u)
        u1 = np.pad(u1, 1, mode = 'symmetric')
        v1 = np.pad(v1, 1, mode = 'symmetric')
        u = np.pad(u, 1, mode = 'symmetric')
        v = np.pad(v, 1, mode = 'symmetric')     
             
        # Neighbourhood size (in Laplacian approximation)
        nb_size = 4 
        factor = (h ** 2) / alpha  
        
        omega = 1     # omega \in (0, 2)
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
                
                # SOR: Need to tune omega with line search
                # May be use omega matrix, since blocks of evolving structures have interacting omegas
                u1_prime[i, j] = numr_u / denr_u
                v1_prime[i, j] = numr_v / denr_v
                u1[i, j] = u1[i, j] + omega * (u1_prime[i, j] - u1[i, j])
                v1[i, j] = v1[i, j] + omega * (v1_prime[i, j] - v1[i, j])
                
        # Remove dummy boundaries
        u = u[1:-1, 1:-1]
        v = v[1:-1, 1:-1]
        u1 = u1[1:-1, 1:-1]
        v1 = v1[1:-1, 1:-1]
        return u1, v1
        
        
    def gauss_seidel_iso(self, u, v, J, alpha = 500, h = 1):
        """ Gauss-Seider Solver for Non-linear system
            Args:
        ----------------
                u: Previous flow fields in horizontal direction
                v: Previous flow fields in vertical direction
                J: Motion tensor
                alpha: Smoothness paramter
                h: Grid size
                
            Returns:
        ----------------
                Returns the computed flow vectors:
                u1: Flow fields in horizontal direction
                v1: Flow fields in vertical direction
        """
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
        
        # sum_small_u = conv2d(u1, small_nb)
        # sum_big_u = conv2d(u, big_nb)
        # sum_small_v = conv2d(v1, small_nb)
        # sum_big_v = conv2d(v, big_nb)
        # numr_u = sum_small_u + sum_big_u - factor * (J12 * v + J13)
        # denr_u = nb_size + factor * J11
        # numr_v  = sum_small_v + sum_big_u - factor * (J12 * v + J23)
        # denr_v = nb_size + factor * J22
        # u1 = numr_u / denr_u
        # v1 = numr_v / denr_v
        
        # TODO: Replace smoothness term (change discretization for div of diffusivity)
        # now it is simple laplacian due to norm of grad square
             
        # Reflecting bc
        u1 = np.zeros_like(u)       
        v1 = np.zeros_like(u)
        u1 = np.pad(u1, 1, mode = 'symmetric')
        v1 = np.pad(v1, 1, mode = 'symmetric')
        u = np.pad(u, 1, mode = 'symmetric')
        v = np.pad(v, 1, mode = 'symmetric')     
             
        # Neighbourhood size (in Laplacian approximation)
        nb_size = 4 
        factor = (h ** 2) / alpha  
        for i in range(1, nrows):
            for j in range(1, ncols):
                sum_small_u = u1[i-1, j] * 0.5 * (g_u[i-1, j] + g_u[i, j]) + u1[i, j-1] * 0.5 * (g_u[i, j-1] + g_u[i, j])
                sum_small_v = v1[i-1, j] * 0.5 * (g_v[i-1, j] + g_v[i, j]) + v1[i, j-1] * 0.5 * (g_v[i, j-1] + g_v[i, j])

                sum_big_u = u1[i+1, j] * 0.5 * (g_u[i+1, j] + g_u[i, j]) + u1[i, j+1] * 0.5 * (g_u[i, j+1] + g_u[i, j])
                sum_big_v = v1[i+1, j] * 0.5 * (g_v[i+1, j] + g_v[i, j]) + v1[i, j+1] * 0.5 * (g_v[i, j+1] + g_v[i, j])
            
                sum_small_u = np.sum((u1[i-1:i+2, j-1:j+2])[small_nb])
                sum_small_v = np.sum((v1[i-1:i+2, j-1:j+2])[small_nb])
                sum_big_u =  np.sum((u[i-1:i+2, j-1:j+2])[big_nb])
                sum_big_v =  np.sum((v[i-1:i+2, j-1:j+2])[big_nb])
                
                numr_u = sum_small_u + sum_big_u - factor * (J12[i, j] * v[i, j] + J13[i, j])
                denr_u = nb_size + factor * J11[i, j]
                numr_v  = sum_small_v + sum_big_u - factor * (J12[i, j] * u[i, j] + J23[i, j])
                denr_v = nb_size + factor * J22[i, j]
                
                u1[i, j] = numr_u / denr_u
                v1[i, j] = numr_v / denr_v
                
        # Remove dummy boundaries
        u = u[1:-1, 1:-1]
        v = v[1:-1, 1:-1]
        u1 = u1[1:-1, 1:-1]
        v1 = v1[1:-1, 1:-1]
        return u1, v1
    
