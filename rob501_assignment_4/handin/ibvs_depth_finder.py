import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)
    
    #--- FILL ME IN ---
    # Calculate point velocity (change in position) and flatten it to a vector
    point_velocity = pts_obs - pts_prev
    point_velocity_flat = point_velocity.flatten(order='F').reshape(-1, 1)

    # Split camera velocity into translational and rotational components
    tv, wv = np.split(v_cam, 2)

    # Compute Jacobians for all points and concatenate into one matrix
    for i in range(n):
        pi = pts_obs[:, i:i+1]
        J_i = ibvs_jacobian(K, pi, 1)
        J[2*i:2*(i+1), :] = J_i

    # Calculate components for depth estimation
    J_t, J_w = np.hsplit(J, 2)
    t = J_t @ tv
    w = point_velocity_flat - J_w @ wv

    # Estimate depths for each point
    for i in range(0, len(t), 2):
        t_slice = t[i:i+2].reshape(-1, 1)
        w_slice = w[i:i+2].reshape(-1, 1)
        least_squares_result = np.linalg.lstsq(t_slice, w_slice, rcond=None)[0]
        depth_estimate = least_squares_result[0, 0]
        zs_est[i//2] = 1.0 / depth_estimate

    # Convert list of depths to numpy array and ensure correct type
    zs_est = np.array(zs_est, dtype=np.float64)
    #------------------
    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est


import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)
    
    #--- FILL ME IN ---
    # Calculate optical flow (change in position) and reshape to a column vector
    point_velocity = pts_obs - pts_prev
    point_velocity_flat = point_velocity.flatten(order='F').reshape(-1, 1)

    # Decompose camera velocity into translational (tv) and rotational (wv) components
    tv, wv = np.split(v_cam, 2)

    # Build the full Jacobian matrix for all image points
    for i in range(n):
        pi = pts_obs[:, i:i+1]
        J_i = ibvs_jacobian(K, pi, 1)  # Compute Jacobian with an assumed depth of 1
        J[2*i:2*(i+1), :] = J_i  # Concatenate the Jacobian for this point

    # Split the full Jacobian into translational (J_t) and rotational (J_w) components
    J_t, J_w = np.hsplit(J, 2)

    # Compute the components of optical flow due to translational and rotational motion
    t = J_t @ tv  # Optical flow component due to translation
    w = point_velocity_flat - J_w @ wv  # Optical flow component after removing rotation effect

    # Solve for the depth of each point using least squares estimation
    for i in range(0, len(t), 2):
        t_slice = t[i:i+2].reshape(-1, 1)
        w_slice = w[i:i+2].reshape(-1, 1)
        least_squares_result = np.linalg.lstsq(t_slice, w_slice, rcond=None)[0]
        depth_estimate = least_squares_result[0, 0]  # First element of the result
        zs_est[i//2] = 1.0 / depth_estimate  # Inverse to get depth

    # Convert list of depths to numpy array and ensure correct type
    zs_est = np.array(zs_est, dtype=np.float64)
    #------------------
    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est

