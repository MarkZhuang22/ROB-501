import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    # Extract the focal length from the camera intrinsic matrix K
    f = K[0, 0] 

    # Compute the normalized image coordinates
    u, v = pt.flatten()
    u_hat = u - K[0, 2]
    v_hat = v - K[1, 2]

    # Compute the Jacobian matrix using the normalized coordinates and the depth
    J = np.array([
        [-f/z, 0, u_hat/z, u_hat*v_hat/f, -(f**2 + u_hat**2)/f, v_hat],
        [0, -f/z, v_hat/z, (f**2 + v_hat**2)/f, -u_hat*v_hat/f, -u_hat]
    ], dtype=np.float64)
    #------------------
    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J