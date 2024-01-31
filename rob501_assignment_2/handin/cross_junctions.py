import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.
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

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    # Initialize matrix A
    A = np.zeros((8, 9))
    
    # Construct the matrix A using the point correspondences
    for i in range(4):
        x, y = I1pts[:, i]
        u, v = I2pts[:, i]
        # Four 2×9 Ai matrices 
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    
    # Compute the null space of A to get h
    h = null_space(A).flatten()
    
    # Reshape h to get the homography matrix H
    H = h.reshape(3, 3)
    
    # Normalize H so that its bottom right value is 1
    H = H / H[2, 2]
    #------------------

    return H, A

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    # Initialize the number of points and the result array
    num_pts = Wpts.shape[1]
    Ipts = np.zeros((2, num_pts), dtype=np.float64)

    # Compute the size of a single box in the grid (assuming uniform grid)
    box_diff = Wpts[0, 1] - Wpts[0, 0]

    # Adjustments for bounding box corners
    x_adjustments = np.array([-1.5, 1.5, 1.5, -1.5]) * box_diff
    y_adjustments = np.array([-1.25, -1.25, 1.25, 1.25]) * box_diff

    # Combine x and y adjustments
    bounding_adjustments = np.array([x_adjustments, y_adjustments])

    # Find minimum and maximum x and y points in Wpts
    min_x, max_x = Wpts[0, :].min(), Wpts[0, :].max()
    min_y, max_y = Wpts[1, :].min(), Wpts[1, :].max()

    # Create bounding box using min and max points
    img_box_base = np.array([min_x, max_x, max_x, min_x, min_y, min_y, max_y, max_y]).reshape(2, 4)

    # Apply adjustments to get the extended bounding box
    img_box = img_box_base + bounding_adjustments

    # Compute the initial homography using DLT
    H, _ = dlt_homography(img_box, bpoly)

    # Add a homogeneous coordinate to world points
    Wpts[2] = 1

    # Use the homography to project the world points into initial image points
    Ijunctions = np.dot(H, Wpts)
    Ijunctions /= Ijunctions[2, :]

    # Round the projected points to get initial integer coordinates
    Ijunctions = np.round(Ijunctions[:2]).astype(int).T

    # Initialize a window size for the saddle point refinement
    window_size = 15

    # Refine each projected point using saddle point detection
    for i, coord in enumerate(Ijunctions):
        # Define the image patch around each projected point
        x_range = slice(coord[0] - window_size, coord[0] + window_size + 1)
        y_range = slice(coord[1] - window_size, coord[1] + window_size + 1)
        
        # Extract the image patch
        img_patch = I[y_range, x_range]
        
        # Find the refined coordinates within the patch using saddle point
        junction_offset = saddle_point(img_patch).flatten().T
        
        # Calculate the refined image point coordinates
        Ipts[:, i] = junction_offset + coord - window_size

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts