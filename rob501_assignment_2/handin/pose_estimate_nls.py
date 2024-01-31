import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    # Some hints on structure are included below...
    # 1. Convert initial guess to parameter vector (6 x 1).
    # Convert the initial 4x4 pose matrix to a 6x1 parameter vector.
    params = epose_from_hpose(Twc_guess)

    # Initialize previous parameters for convergence check
    params_prev = np.zeros((6,1))
    iter = 1  

    # 2. Main loop - continue until convergence or maxIters.
    while True:
        # 3. Save previous best pose estimate.
        # Store the previous parameter values to check for convergence later.
        params_prev = params.copy()
        
        # Extract rotation matrix and translation vector from current parameters
        C_current = dcm_from_rpy(params[3:6])
        t_current = params[:3]
        
        # 4. Project each landmark into image, given current pose estimate.
        for i in np.arange(tp):
            Wpt = Wpts[:, i:i+1]  # Extract world point
            Ipt_actual = Ipts[:, i:i+1]  # Extract corresponding image point
            
            # Perform perspective projection using the current pose
            Ipt_projected_homogeneous = K @ C_current.T @ (Wpt - t_current)
            Ipt_projected = (Ipt_projected_homogeneous[:2] / Ipt_projected_homogeneous[2])[0:2, :]
            
            # Calculate the residual for this point
            dY[2*i:2*i+2, :] = Ipt_projected - Ipt_actual

        # Compute Jacobian for each point
        for i in np.arange(tp):
            Wpt = Wpts[:, i:i+1]
            J_i = find_jacobian(K, hpose_from_epose(params), Wpt)
            J[2*i:2*i+2, :] = J_i

        # 5. Solve system of normal equations for this iteration.
        # Solve for the parameter update using least squares
        d_params = np.linalg.lstsq(J.T @ J, -J.T @ dY, rcond=None)[0]
        params += d_params  # Update the parameters

        # 6. Check - converged?
        diff = norm(params - params_prev)

        if norm(diff) < 1e-12:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
                
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    Twc = hpose_from_epose(params)

    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc