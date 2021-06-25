import os
import numpy as np
from PIL import Image
from time import time
from scipy import ndimage

class TvInPaint():
    def __init__(self):
        self.u_solver = self.gauss_seidel
        self.d_solver = self.shrinkage
        
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
    
    def divergence(self, f, hx = 1.0, hy = 1.0, bc = 0):
        f1 = np.array(f[..., 0], copy = True)
        f2 = np.array(f[..., 1], copy = True)
        
        curr_1x = f1
        curr_2y = f2
        # Shift forward/down by 1 to get f1_{i-1, j}/f2_{i, j-1}
        prev_1x = np.roll(f1, 1, axis = 1)
        prev_2y = np.roll(f2, 1, axis = 0)
        
        # Divergence matrix is negative transpose of forward difference matrix
        prev_1x[:, 0] =  0
        prev_2y[0, :] =  0
        if bc in [0, 'neumann']:
            # Reflecting boundary conditions
            curr_1x[:, -1] =  0
            curr_2y[-1, :] =  0
        elif bc in [1, 'dirichlet']:
            # Dirichlet boundary conditions
            # satisfied automatically
            pass
            
        f_1x = (curr_1x - prev_1x) / hx
        f_2y = (curr_2y - prev_2y) / hy
        return f_1x + f_2y
    
    def gauss_seidel(self, u, f, d, b, lambd, gamma):
        h, w = np.shape(u)[:2]
        
        alpha = lambd / gamma
        small_nb = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 0]], dtype = bool)
        big_nb = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]], dtype = bool)
        
        t1 = time()
        
        c = d - b
        # Divergence computation
        div_c = self.divergence(c)
         
        # Neighbourhood size 
        nb_size = np.ones((h, w)) * 4
        # Corner pixels
        nb_size[0, 0] = nb_size[0, w-1] = nb_size[h-1, 0] = nb_size[h-1, w-1] = 2
        # Boundary pixels
        nb_size[0, 1:h-1] = nb_size[1:w-1, 0] = 3

        t2 = time()
        
        # Use correlation operation for fast sum compute
        big_nb = big_nb[..., np.newaxis]
        sum_big_u = ndimage.correlate(u, big_nb, mode = 'reflect')
        
        numr_u = alpha[..., None] * f - div_c + sum_big_u
        factor = 1. / (nb_size + alpha)
        
        t3 = time()
        
        u1 = np.zeros([h + 2, w + 2] + list(u.shape[2:]))
        for i in range(h):
            for j in range(w):
                # Adjust to padding
                i1 = i + 1
                j1 = j + 1
                
                # u1 has padding, so: (i1, j1) => (i+1, j+1), (i1-1, j1) => (i, j+1), (i1, j1-1) => (i+1, j)
                u1[i1, j1] = factor[i, j] * (numr_u[i, j] + u1[i1 - 1, j1] + u1[i1, j1 - 1])
        
        t4 = time()
        
        # print("Compute time t1: {0:.4f} \t t2: {1:.4f} \t t3: {2:.4f}".format(t2-t1, t3-t2, t4-t3))
        
        u = u[1:h+1, 1:w+1]
        u1 = u1[1:h+1, 1:w+1]
        
        return u1
        
    def shrinkage(self, u, f, d, b, lambd, gamma):
        thresh = 1. / gamma
        
        ux, uy = self.forward_diff(u)
        nabla_u = np.stack([ux, uy], axis = -1)
        
        e = nabla_u + b
        e_norm = np.linalg.norm(e, axis = -1) + 1e-7
        e_norm = np.stack([e_norm, e_norm], axis = -1)
        e_uv = e / e_norm
        
        d = e_uv * np.maximum(e_norm - thresh, 0)
        b = b + nabla_u - d
        
        return d, b
        
    def init_u(self, f, lambd):
        u = np.array(f, copy = True)
        
        u_gray = u[..., 0] * 0.299 + u[..., 1] * 0.587 + u[..., 2] * 0.114
        pos = (u_gray > 0.5) * (lambd == 0) 
        u[pos] = 0.5
        
        return u

    def inpaint(self, f, lambd = 25, gamma = 5, tol = 1e-5, maxiter = 100, save_itr = False, itr_path = './save/iterates'):
        h, w = f.shape[:2]
        # u is set to the initial guess
        u = self.init_u(f, lambd)
        
        # norm is used for convergence threshold
        f_norm = np.linalg.norm(f)
        
        d = np.zeros(list(u.shape) +  [2], dtype = np.float)
        b = np.zeros(list(u.shape) +  [2], dtype = np.float)
        
        diff_norm = 1000 * tol
        
        for it in range(maxiter):
            t1 = time()
            d, b = self.d_solver(u, f, d, b, lambd, gamma)
            t2 = time()
            u1 = self.u_solver(u, f, d, b, lambd, gamma)
            t3 = time()
            
            print("Time taken by: d-solver: {0:.4f} \t u-solver: {1:.4f}".format(t2-t1, t3-t2))
            
            diff_norm = np.linalg.norm(u1 - u) / f_norm
            u = np.clip(u1, 0, 1)
            
            print("Iteration: {0} \t Distance: {1:0.6f}".format(it, diff_norm))
            if it > 2 and diff_norm  < tol:
                break
                
            if save_itr == True:
                masked_img = Image.fromarray((255 * u).astype(np.uint8))
                masked_img.save(os.path.join(itr_path, 'itr-{}.png'.format(it)))
                # masked_img.show()
        return u

def get_mask(f, display = False):
    mask = (f == 255).all(axis = -1)
    if display:
        mask_3d = np.repeat(mask[..., np.newaxis], 3, axis = -1).astype(np.int)
        # Keep only blue channel in identified mask locations
        mask_3d[..., :2] *= 255
        masked_arr =  f - mask_3d
        masked_img = Image.fromarray(masked_arr.astype(np.uint8))
        masked_img.show()
    return mask   

def main():
    kmax = 50       # Number of iterations              
    lambd = 50      # Fidelty weight
    tol = 1e-3      # Convergence tolerance
    gamma = 5       # Penalty weight on constraint d = grad u
    gamma2 = 8      # Penalty weight on constraint z = u
    
    save_flag = True
    
    load_path = os.path.abspath('./input/')
    save_path = os.path.abspath('./save/')
    itr_path = os.path.abspath('./save/iterates')
    assert os.path.exists(load_path), 'The path {} does not exist'.format(load_path)
    
    if not os.path.exists(itr_path) and save_flag:
        os.makedirs(itr_path)
    
    # print('Available input files in directory ./{0}'.format(os.path.basename(load_path)))
    available_files = os.listdir(load_path)
    for file_name in available_files:
        print(file_name)
    
    in_path = input('Enter image path: ')
    # in_path = os.path.join(load_path, 'car_text.png')
    filename = os.path.basename(in_path).split('.')[0]
    
    image = Image.open(in_path)
    f = np.array(image, dtype = np.float)
    
    print("Image {0} loaded...".format(filename))
    
    # Create space varying lambd that is 0 in region to be inpainted
    mask = get_mask(f, display = False)
    lambd = (1 - mask) * lambd
    
    f /= 255.
    tv = TvInPaint()
    f_out = tv.inpaint(f, lambd = lambd, gamma = gamma, tol = tol, maxiter = kmax, save_itr = False, itr_path = itr_path)
    f_out *= 255.
    
    image_out = Image.fromarray(f_out.astype(np.uint8))
    savename = filename + '_inpainted.png'
    out_path = os.path.join(save_path, savename)
    
    if save_flag:
        image_out.save(out_path)
        
    image_out.show()


if __name__ == '__main__':
    main()
