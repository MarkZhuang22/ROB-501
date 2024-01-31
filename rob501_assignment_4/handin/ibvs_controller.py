import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """
    v = np.zeros((6, 1))

    #--- FILL ME IN ---
    # Compute the error between the desired and observed image points
    error = pts_des - pts_obs
    # Flatten the error in column-major order
    error_vec = error.flatten('F') 

    # Initialize the stacked image Jacobian matrix
    J_stacked = np.zeros((2 * pts_obs.shape[1], 6))

    # Compute the stacked image Jacobian for all points
    for i in range(pts_obs.shape[1]):
        J_stacked[2*i:2*i+2, :] = ibvs_jacobian(K, pts_obs[:, i], zs[i])

    # Compute the desired feature velocities (proportional control law)
    feature_velocities = gain * error_vec.flatten('F')

    # Compute the camera velocity using the pseudo-inverse of the stacked Jacobian
    v = np.linalg.pinv(J_stacked) @ feature_velocities

    # Reshape v to ensure it's a 6x1 vector
    v = v.reshape((6, 1))
    #------------------
    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return v