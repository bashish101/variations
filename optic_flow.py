import numpy as np
from PIL import Image

from solvers import Solver
from image_op import get_derivatives

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
        
    def compute_flow(self, f1, f2, solver = 'explicit', alpha = 500, lambd = 4, tau = 0.2, maxiter = 10000, grid_steps = [2, 4, 8]):
        r, c = f1.shape[:2]
        u = v = np.zeros((r, c))
        fx, fy, ft = get_derivatives(f1, f2)
        print('Alpha:', alpha)
        print('Lambda:', lambd)
        print('tau:', tau)
        if solver == 'explicit':
            for it in range(maxiter):
                print('Iteration-', it)
                u, v = self.solver.explicit_solver(fx, fy, ft, u, v, alpha = alpha, lambd = lambd, tau = tau)
                mag = np.sqrt(u ** 2 + v ** 2)
                
                print('Max: {0:.2f} Min: {1:.2f} Mean: {2:.2f} std: {3:.2f}'.format(np.amax(mag), np.amin(mag), np.mean(mag), np.std(mag)))
        else:
            u, v = self.solver.multi_grid_solver(f1, f2, grid_steps = grid_steps, alpha = alpha)
            mag = np.sqrt(u ** 2 + v ** 2)
            print('Max mag: {0:.2f} Mean mag: {1:.2f}'.format(np.amax(mag), np.mean(mag)))
        
        vis = self.visualize(u, v)
        return vis
        
    
def main():
    kmax = 100              # Number of iterations (we are interested in steady state of the diffusion-reaction system)
    alpha = 500             # Regularization Parameter (should be large enough to weight smoothness terms which have small magnitude)
    tau = 0.2               # Step size (For implicit scheme, can choose arbitrarily large, for explicit scheme  <=0.25)
    lambd = 0
    frame1_path = input('Enter first image: ')
    frame2_path = input('Enter second image: ')
    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    
    f1 = np.array(frame1, dtype = np.float)
    f2 = np.array(frame2, dtype = np.float)
    
    optic_flow = OpticFlow()
    vis = optic_flow.compute_flow(f1, f2, alpha = alpha, lambd = lambd, tau = tau, maxiter = kmax)
    
    vis = Image.fromarray(vis)
    vis.save('flow_visual.jpg')
    vis.show()

if __name__ == '__main__':
    main()    
    
