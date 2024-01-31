import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    #    # Setting up the A matrix and b vector for the linear least squares fitting
    shape_y, shape_x = I.shape  # Dimensions of the image patch
    A = np.empty((shape_y * shape_x, 6))
    b = np.empty((shape_y * shape_x, 1))
    
    i = 0
    for y in range(0, shape_y):
        for x in range(0, shape_x):
            A[i, :] = [x ** 2, x * y, y ** 2, x, y, 1]  # Corresponds to Eqn. (4) in the paper
            b[i] = I[y, x]  # Intensity value at the point (x, y)
            i += 1

    # Solving the linear least squares problem to get the parameters
    params, *_ = lstsq(A, b, rcond=None)
    alpha, beta, gamma, delta, epsilon, zeta = params.flatten()
    
    # Calculating the saddle point coordinates (x, y)
    M = np.array([[2 * alpha, beta], [beta, 2 * gamma]])  
    x = np.array([[delta], [epsilon]]) 
    pt = -np.matmul(inv(M), x)  # Saddle point coordinates

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt