import numpy as np

def forward_diff(f, hx = 1.0, hy = 1.0, bc = 0):        
    """ Computes gradient (∇_{+}f) using forward difference approximation
    f_x = (f_{i+1, j} - f{i, j}) / h_x
    f_y = (f_{i, j+1} - f{i, j}) / h_y
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            Returns the following derivatives:
            fx: Forward difference approximation in x-direction
            fy: Forward difference approximation in y-direction
        This approximation is of consistency order O(h)
    """
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

def backward_diff(f, hx = 1.0, hy = 1.0, bc = 0):        
    """ Computes gradient (∇_{-}f) using backward difference approximation
    f_x = (f_{i+1, j} - f{i-1, j}) / h_x
    f_y = (f_{i, j+1} - f{i, j-1}) / h_y
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            Returns the following derivatives:
            fx: Backward difference approximation in x-direction
            fy: Backward difference approximation in y-direction
        This approximation is of consistency order O(h)
    """
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
    
def central_diff(f, hx = 1.0, hy = 1.0, bc = 0):        
    """ Computes gradient (∇_{0}f) using central difference approximation
    f_x = (f_{i+1, j} - f{i, j}) / (2 * h_x)
    f_y = (f_{i, j+1} - f{i, j}) / (2 * h_y)
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            Returns the following derivatives:
            fx: Central difference approximation in x-direction
            fy: Central difference approximation in y-direction
        This approximation is of consistency order O(h^2)
    """
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

def divergence(f, hx = 1.0, hy = 1.0, bc = 0):
    """ Computes divergence (∇.f) as transposed gradient operation
    f_x = (f_{i+1, j} - f{i, j}) / (2 * h_x)
    f_y = (f_{i, j+1} - f{i, j}) / (2 * h_y)
      ∇.f = f_x + f_y
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            div f: Divergence of f
    """
    
    curr_x = np.array(f, copy = True)
    curr_y = np.array(f, copy = True)
    # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
    prev_x = np.roll(f, 1, axis = 1)
    prev_y = np.roll(f, 1, axis = 0)
    
    # Divergence matrix is negative transpose of forward difference matrix
    prev_x[:, 0] =  0
    prev_y[0, :] =  0
    if bc in [0, 'neumann']:
        # Reflecting boundary conditions
        curr_x[:, -1] =  0
        curr_y[-1, :] =  0
    elif bc in [1, 'dirichlet']:
        # Dirichlet boundary conditions
        # satisfied automatically
        pass
        
    fx = (curr_x - prev_x) / hx
    fy = (curr_y - prev_y) / hy
    return fx + fy

def curl(f, hx = 1.0, hy = 1.0, bc = 0):
    """ Computes curl (∇×f) in 2d using central-difference approximations
    f_x = (f_{i+1, j} - f_{i-1, j}) / (2 * h_x)
    f_y = (f_{i, j+1} - f_{i, j-1}) / (2 * h_y)
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            div f: Divergence of f
        This approximation is of consistency order O(h^2)
    """
    fx_c, fy_c = central_diff(f, hx, hy, bc)
    return fx_c - fy_c
    
def laplacian(f, hx = 1.0, hy = 1.0, bc = 0):     
    """ Computes laplacian (Δf) using half-pixel grid-sizes
    (or alternatively, forward and backward difference combination)
    f_xx = (f_{i+1, j} - 2 * f{i, j} + f_{i-1, j}) / (h_x^2)
    f_yy = (f_{i, j+1} - 2 * f{i, j} + f_{i, j-1}) / (h_y^2)
    Δf = f_xx + f_yy
    (Uses Homogeneous Neumann boundary conditions)
        Args:
    ----------------
            f: Input image
            bc: Boundary condition 
                0 for Homogeneous Neumann Boundary condition
                1 for Dirichlet Boundary condition
        Returns:
    ----------------
            Delta f: Laplacian of f
        This approximation is of consistency order O(tau + h_x^2 + h_y^2)
    """
    # Shift back/above by 1 to get f_{i+1, j}/f_{i, j+1}
    next_x = np.roll(f, -1, axis = 1)
    next_y = np.roll(f, -1, axis = 0)
    
    # Shift forward/down by 1 to get f_{i-1, j}/f_{i, j-1}
    prev_x = np.roll(f, 1, axis = 1)
    prev_y = np.roll(f, 1, axis = 0)
    
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
        
    f_xx = (next_x - 2 * f + prev_x) / (hx ** 2)
    f_yy = (next_y - 2 * f + prev_y) / (hy ** 2)
    return f_xx + f_yy

def get_derivatives(f1, f2, hx = 1.0, hy = 1.0, ht = 1.0):
    """ Computes spatial and temporal derivatives for the given frames
        Args:
    ----------------
            f1: First frame
            f2: Second frame
        Returns:
    ----------------
            Returns the following derivatives:
            fx: Spatial derivative in x-direction
            fy: Spatial derivative in y-direction
            ft: Temporal derivative
    """
    fx1, fy1 = central_diff(f1, hx, hy) 
    fx2, fy2 = central_diff(f2, hx, hy)
    fx = (fx1 + fx2) / 2.0
    fy = (fy1 + fy2) / 2.0
    ft = (f2 - f1) / ht
    return fx, fy, ft
    
def gauss_conv(f, sigma, precision = 3, warn = False):        
    """ Computes 2D Gaussian convolution using separable 1D convolutions
        Args:
    ----------------
            f: Input image
            sigma: Standard deviation for Gaussian
            precision: Desired precision of approximation (Determines size of kernel)
        Returns:
    ----------------
            Returns the following derivatives:
            f_conv: Convolution of f with Gaussian kernel
    """
    # Center index for symmetric weights, array indexing starts from 0
    f_size = len(f.shape)
    if f_size == 2:
        f = f.reshape((f.shape[0], f.shape[1], 1))
    rows, cols, ch = f.shape
    center = int(precision * sigma)     
    pad_length = 2 * center
    wt_size = pad_length + 1

    if warn and (wt_size > rows or wt_size > cols):
        print('Warning! Kernel size exceeds signal length!')

    # Gaussian weights
    # factor = 1 / (sigma * np.sqrt(2. * np.pi))
    wts = np.array([np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) for x in range(wt_size)]) 
    wts /= sum(wts)

    # Horizontal padding with reflecting boundary conditions
    if cols < center:
        f_ext = np.pad(f, [(0,), (center,), (0,)], mode = 'symmetric')
    else:
        f_ext = np.zeros((rows, cols + pad_length, ch))
        f_ext[:, center:-center] = f
        f_ext[:, center-1::-1] = f[:, 0:center]
        f_ext[:, -center:] = f[:, -1:-center-1:-1]

    f_conv = np.zeros((rows, cols, ch))
    # Convolve horizontally
    for r in range(rows):
        for c in range(cols):
            f_conv[r, c] = sum([f_ext[r, c + idx] * wts[idx] for idx in range(wt_size)])

    # Vertical padding with reflecting boundary conditions
    if rows < center:
        f_ext = np.pad(f_conv, [(center,), (0,), (0,)], mode = 'symmetric')
    else:
        f_ext = np.zeros((rows + pad_length, cols, ch))
        f_ext[center:-center] = f_conv
        f_ext[center-1::-1] = f_conv[0:center]
        f_ext[-center:] = f_conv[-1:-center-1:-1]


    # Convolve vertically
    for r in range(rows):
        for c in range(cols):
            f_conv[r, c] = sum([f_ext[r + idx, c] * wts[idx] for idx in range(wt_size)])
    if f_size == 2:
        f_conv = f_conv.reshape((f_conv.shape[0], f_conv.shape[1]))
        
    return f_conv

def fft_gauss(f, sigma, precision = 3, warn = False):
    """ Computes 2D Gaussian convolution using FFT
        Args:
    ----------------
            f: Input image
            sigma: Standard deviation for Gaussian
            precision: Desired precision of approximation (Determines size of kernel)
        Returns:
    ----------------
            Returns the following derivatives:
            f_conv: Convolution of f with Gaussian kernel
    """
    center = precision * sigma
    length = 2 * precision * sigma + 1
    if warn and (length > f.shape[0] or length > f.shape[1]):
        print('Warning! Kernel size exceeds the signal length! Convolution can cause approximation error.')

    kernel = np.linspace(-center, center, length)
    xx, yy = np.meshgrid(kernel, kernel)

    factor = 1 / (sigma * np.sqrt(2. * np.pi))
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    return conv2d(f, kernel)

def conv2d(f, kernel):
    """ Computes 2D convolution with reflecting bc using FFT
        Args:
    ----------------
            f: Input image
            kernel: Kernel to be convolved on f
        Returns:
    ----------------
            Returns the following derivatives:
            f_conv: Convolution of f with kernel
    """
    r_f, c_f = f.shape[:2]
    r_k, c_k = kernel.shape[:2]

    top_pad = r_k // 2
    bottom_pad = r_f + r_k // 2
    left_pad = c_k // 2
    right_pad = c_f + c_k // 2

    # Make signal symmetric around boundaries
    f = np.pad(f, [(top_pad, bottom_pad), (left_pad, right_pad)], mode = 'symmetric')
    fr_f = np.fft.fft2(f)
    fr_k = np.fft.fft2(kernel, s = f.shape)
    fr_conv = fr_f * fr_k
    f_conv = np.real(np.fft.ifft2(fr_conv))

    top = r_k - 1
    bottom = top + r_f 
    left = c_k - 1
    right = left + c_f

    f_conv = f_conv[top:bottom, left:right]

    return f_conv 

def downsample2d(f, rate):
    if rate == 1:
        return f
    return f[::rate, ::rate]
    
def upsample2d(f, rate):
    if rate == 1:
        return f
    f = np.repeat(f, rate, axis = 0)
    f = np.repeat(f, rate, axis = 1)
    return f
           
