import numpy as np
from PIL import Image
from skimage.filters import gaussian as gaussian_filter

from image_op import *
from time import time
from multigrid import full_multigrid

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Solver(nn.Module):
    def __init__(self,
                 num_cycles = 3):
        self.num_cycles = num_cycles
        self.max_depths = [3] * self.num_cycles
        jacobi_kernel = torch.tensor([[0., 1., 0.],
                                     [1., 0., 1.],
                                     [0., 1., 0.]])
        self.jacobi_kernel = jacobi_kernel.view(1, 1, 3, 3)
        
        
        laplace_kernel = torch.tensor([[0., 1., 0.],
                                       [1., -4, 1.],
                                       [0., 1., 0.]])
        self.laplace_kernel = laplace_kernel.view(1, 1, 3, 3)
        self.base_solver = None
        
    def central_diff(self, f, hx = 1.0, hy = 1.0, bc = 0):        
        # Shift back/above by 1 to get f_{i+1, j}/f_{i, j+1}
        next_x = torch.roll(f, -1, dims = -1)
        next_y = torch.roll(f, -1, dims = -2)

        # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
        prev_x = torch.roll(f, 1, dims = -1)
        prev_y = torch.roll(f, 1, dims = -2)

        # Reflecting boundary conditions
        if bc in [0, 'neumann']:
            # Reflecting boundary conditions
            next_x[..., -1] =  next_x[..., -2]
            next_y[:, :, -1, :] =  next_y[:, :, -2, :]
            prev_x[..., 0] =  prev_x[..., 1]
            prev_y[:, :, 0, :] =  prev_y[:, :, 1, :]
        elif bc in [1, 'dirichlet']:
            # Dirichlet boundary conditions
            next_x[..., -1] =  0
            next_y[:, :, -1, :] =  0
            prev_x[..., 0] =  0
            prev_y[:, :, 0, :] =  0

        fx = (next_x - prev_x) / (2 * hx)
        fy = (next_y - prev_y) / (2 * hy)
        return fx, fy
    
    def compute_gradient(self, f1, f2):
        fx1, fy1 = self.central_diff(f1)   
        fx2, fy2 = self.central_diff(f2)   
        
        nabla_fx = (fx1 + fx2) / 2.0
        nabla_fx = (fy1 + fy2) / 2.0
        nabla_ft = (f2 - f1) / ht
        return nabla_fx, nabla_fy, nabla_ft
    
    def compute_struct_tensor(self, f1, f2, hx = 1.0, hy = 1.0, ht = 1.0, rho = 5, sigma = 0.5):
        bs, ch, h, w = f1.shape
        inner_filter = GaussianSmoothing(sigma = sigma, dim = 2)
        outer_filter =  GaussianSmoothing(sigma = rho, dim = 2)
        
        f1 = inner_filter(f1)
        f2 = inner_filter(f2)
        
        nabla_fx, nabla_fy, nabla_ft = self.compute_gradient(f1, f2)
        
        t11 = nabla_fx * nabla_fx
        t12 = nabla_fx * nabla_fy
        t13 = nabla_fx * nabla_ft
        t22 = nabla_fy * nabla_fy
        t23 = nabla_fy * nabla_ft
        t33 = nabla_ft * nabla_ft
        
        J = torch.stack([t11, t12, t13, t22, t23, t33], dim = 1).view(bs, -1, h, w) # shape: bs, 6, 3, h, w
        J = outer_filter(J)
        J = J.view(bs, 6, ch, h, w).sum(dim = 2)                                    # shape: bs, 6, h, w
        
        return (J[:, 0], J[:, 1], J[:, 2], J[:, 3], J[:, 4])
        
    
    def compute_struct_tensor_old(self, f1, f2, hx = 1.0, hy = 1.0, ht = 1.0, rho = 5, sigma = 0.5):
        # Compute structure tensor with smoothing in integration scale
        # rho >= 2 * sigma
        
        h, w = f1.shape[-2:]
        # f1 = f1.reshape(h, w, -1)                                       # shape = (h, w, ch)
        # f2 = f2.reshape(h, w, -1)                                       # shape = (h, w, ch)
        
        f1 = np.stack([gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = sigma, multichannel=True) for im in f1], dim = 0) # shape = (bs, h, w, ch)
        f1 = torch.from_numpy(f1).permute(0, 3, 1, 2) # shape = (bs, ch, h, w)
        fx1, fy1 = self.central_diff(f1)   
        
        f2 = np.stack([gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = sigma, multichannel=True) for im in f2], dim = 0)
        f2 = torch.from_numpy(f2).permute(0, 3, 1, 2)
        fx2, fy2 = self.central_diff(f2)   
        
        nabla_fx = (fx1 + fx2) / 2.0
        nabla_fx = (fy1 + fy2) / 2.0
        nabla_ft = (f2 - f1) / ht
        
        t11 = nabla_fx * nabla_fx
        t12 = nabla_fx * nabla_fy
        t13 = nabla_fx * nabla_ft
        t22 = nabla_fy * nabla_fy
        t23 = nabla_fy * nabla_ft
        t33 = nabla_ft * nabla_ft
        
        # apply Gaussian filter separately to each (h, w, ch) image and add across channels
        J11 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1) for im in t11])  
        J12 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1)  for im in t12])
        J13 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1)  for im in t13])
        J22 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1) for im in t22])
        J23 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1)  for im in t23])
        J33 = torch.tensor([np.sum(gaussian_filter(im.permute(1, 2, 0).numpy(), sigma = rho, multichannel=True), axis = -1)  for im in t33])

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
        
    def get_residual(self, u, v, J, alpha, h):
        # Computes residual: 
        # r^h = f^h - A^h x_tilde^h
        # This residual computation is for Laplacian in smoothness term E-L
        
        (J11, J12, J13, J22, J23) = J
        
        factor = (h * h) / alpha 
                              
        f = factor * (J12 * v + J13)
        A_u = F.conv2d(u, self.laplace_kernel) + factor * J11 * u
        
        res_u = f - A_u
        
        g = factor * (J12 * u + J23)
        A_v = F.conv2d(v, self.laplace_kernel) + factor * J22 * v
        
        res_v = g - A_v
        
        return res_u, res_v
    
    def cycle(self, u, v, J, depth, max_depth, alpha = 500, hx = 1, hy = 1, smoothiter = 2):
        # If scale is coarsest apply time-marching, stop
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
        r_u_down = F.avg_pool2d(r_u, step)
        r_v_down = F.avg_pool2d(r_v, step)
        
        J11_down = F.avg_pool2d(J11, step)
        J12_down = F.avg_pool2d(J12, step)
        J13_down = r_u_down
        J22_down = F.avg_pool2d(J22, step)
        J23_down = r_v_down
        J_down = (J11_down, J12_down, J13_down, J22_down, J23_down)
        
        # Compute errors
        e1, e2 = self.cycle(r_u_down, r_v_down, J_down, depth + 1, max_depth, alpha, 2 * hx, 2 * hy, smoothiter)
        # base_solver(r_u, r_v, J_down, alpha, hx, hy)
        
        # Upsample
        e1 = F.interpolate(e1, scale_factor = step, mode = 'bilinear')
        e2 = F.interpolate(e2, scale_factor = step, mode = 'bilinear')
        
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
            Returns:
        ----------------
                Returns the computed flow vectors:
                u: Flow vector in horizontal direction
                v: Flow vector in vertical direction
        """
            
        compute_time = []
        
        
        t1 = time()
        J = self.compute_struct_tensor(f1, f2, rho = 5)
        t2 = time()
        t_diff = t2 - t1
        compute_time.append(t_diff)
        print('Structure tensor computation: {:.4f}'.format(t_diff))
        
        # Initialization
        height, width = f1.shape[:2]
        u = np.zeros((height, width))
        v = np.zeros((height, width))
        
        self.max_depths = [3] * self.num_cycles
        for idx in range(self.num_cycles):
            # Solve using f_tilde instead of f in a cycle, for accuracy use multiple correcting multigrid cycles
            # f_tilde, g_tilde = self.compute_f_tilde(f, v)
            
            t1 = time()
            u1, v1 = self.cycle(u, v, J, depth = 1, max_depth = self.max_depths[idx], alpha = 500, hx = 1, hy = 1, smoothiter = 2)
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
            sum_nb_u = F.conv2d(u, self.jacobi_kernel)
            sum_nb_v = F.conv2d(v, self.jacobi_kernel)
            
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
    def __init__(self):
        self.solver = Solver()
        
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
        
    def compute_flow(self, f1, f2, alpha = 500, lambd = 4, tau = 0.2, maxiter = 1000, solver = 'explicit', base_solver = 'jacobi'):
        bs, ch, h, w = f1.shape[-2:]
        
        u = v = torch.zeros((bs, h, w), device = device)
        
        if base_solver == 'jacobi':
            self.solver.base_solver = self.solver.jacobi
        else:
            self.solver.base_solver = self.solver.gauss_seidel
            
        print('Alpha:', alpha)
        print('Lambda:', lambd)
        print('tau:', tau)
        
        u, v = self.solver.multi_grid_solver(f1, f2, alpha = alpha)
        
        
        vis = self.visualize(u, v)
        return vis
        
    
def main():
    kmax = 1500              # Number of iterations (we are interested in steady state of the diffusion-reaction system)
    alpha = 500             # Regularization Parameter (should be large enough to weight smoothness terms which have small magnitude)
    tau = 0.2               # Step size (For implicit scheme, can choose arbitrarily large, for explicit scheme  <=0.25)
    lambd = 4
    solver = 'multi_grid'
    base_solver = 'jacobi'
    # frame1_path = input('Enter first image: ')
    # frame2_path = input('Enter second image: ')
    frame1_path = 'a.pgm'
    frame2_path = 'b.pgm'
    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    
    f1 = np.array(frame1, dtype = np.float)
    f1 = np.ascontiguousarray(f1[None].transpose(0, 3, 1, 2))
    f1 = torch.tensor(f1, device = device)
    
    f2 = np.array(frame2, dtype = np.float)
    f1 = np.ascontiguousarray(f2[None].transpose(0, 3, 1, 2))
    f2 = torch.tensor(f2, device = device)
    
    optic_flow = OpticFlow()
    vis = optic_flow.compute_flow(f1, f2, alpha = alpha, lambd = lambd, tau = tau, maxiter = kmax, solver = solver, base_solver = base_solver)
    vis = Image.fromarray(vis)
    vis.save('./visual.pgm')
    vis.show()

if __name__ == '__main__':
    main()    
    
