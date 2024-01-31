import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---
    """
    This stereo matching algorithm employs a Laplacian filter for initial image enhancement, 
    aiming to accentuate edges and improve the subsequent block matching process. The disparity 
    map is computed using the Sum of Absolute Differences (SAD) method, a common approach in 
    stereo vision for its simplicity and efficiency. Padding is applied to handle edge cases 
    effectively. The algorithm follows a traditional block matching pipeline, with additional 
    post-processing steps such as median and percentile filtering to reduce noise and outliers 
    in the disparity map. 

    References:
    - Brown, M., and Lowe, D. G. (2002). Invariant Features from Interest Point Groups. 
    British Machine Vision Conference.
    - Hirschmüller, H. (2008). Stereo Processing by Semiglobal Matching and Mutual Information. 
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2), 328-341.
    """

    # Code goes here...
    # Setting initial disparity map to zeros
    Id = np.zeros(Il.shape, dtype=Il.dtype)  

    # Window size and half window
    win_size = 11
    win_half = win_size // 2
    
    # Laplacian kernel definition
    laplacian_kernel = np.ones((3, 3)) * -1
    laplacian_kernel[1, 1] = 9.5
    
    # Padding and convolution
    Il_padded, Ir_padded = [np.pad(convolve(I, laplacian_kernel, mode='nearest'), maxd, mode='edge') for I in [Il, Ir]]

    # Defining the bounding box limits
    x_min, x_max, y_min, y_max = bbox[0, 0], bbox[0, 1] + 1, bbox[1, 0], bbox[1, 1] + 1

    # Setting up the aggregation kernel
    aggregation_kernel = np.full((win_half, win_half), 1)

    # Full left image region within bounding box    
    left_region_full = Il_padded[y_min + maxd : y_max + maxd, x_min + maxd : x_max + maxd]

    # Preparing an array for SAD values
    sad_values_all = np.zeros((y_max - y_min, x_max - x_min, maxd+1), dtype=Il.dtype)

    # Disparity calculation loop
    for disparity in range(maxd + 1):
        # Right image region shifted by the current disparity
        right_region = Ir_padded[y_min + maxd : y_max + maxd, x_min + maxd - disparity : x_max + maxd - disparity]
        # SAD calculation
        sad_values = convolve(np.abs(left_region_full - right_region), aggregation_kernel, mode='mirror')
        # Storing SAD values
        sad_values_all[:, :, disparity] = sad_values

    # Assigning disparity values
    Id[y_min:y_max, x_min:x_max] = sad_values_all.argmin(axis=2)

    # Applying median and percentile filters
    Id_slice = Id[y_min:y_max, x_min:x_max]
    Id_slice[:] = median_filter(Id_slice, size=13, mode='nearest')
    Id_slice[:] = percentile_filter(Id_slice, percentile=55, size=4, mode='nearest')
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id