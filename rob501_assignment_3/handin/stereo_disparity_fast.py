import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

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

    # Choose a window size
    window_size = 15
    half_window = window_size // 2

    # Pad images to handle borders by replicating edge pixels
    Il_padded = np.pad(Il, half_window, mode='edge')
    Ir_padded = np.pad(Ir, half_window, mode='edge')

    # Initialize the disparity map with zeros
    Id = np.zeros(Il.shape)

    # Loop over rows within the bounding box
    for y in range(bbox[1, 0], bbox[1, 1] + 1):
        # Loop over columns within the bounding box
        for x in range(bbox[0, 0], bbox[0, 1] + 1):
            min_sad = np.inf
            best_disparity = 0

            # restrict the range to positive disparities
            for d in range(0, maxd + 1):
                # Ensure we stay within the right image boundaries
                if x - d >= 0:
                    # Compute SAD
                    sad = np.sum(np.abs(Il_padded[y:y + window_size, x:x + window_size] -
                                        Ir_padded[y:y + window_size, x - d:x - d + window_size]))
                    # Update the minimum SAD and the best disparity
                    if sad < min_sad:
                        min_sad = sad
                        best_disparity = d

            # Assign the best disparity to the disparity map
            Id[y, x] = best_disparity

    # Post-processing with median filtering to remove noise and outliers
    Id = median_filter(Id, size=3)

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id