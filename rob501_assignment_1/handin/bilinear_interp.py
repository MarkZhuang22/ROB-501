import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    
    # Y&D is rgb. must same transform each band
    # I --> image with all coordinate points. (pt[0], pt[1]) corresponds to np.array([_], dtype= np.int8)
    # but right now I is just greyscale so all points only have 1 value
    # (row, col) --> (y, x) maybe... bc image given to us facing us. 
    # [[0,0,0]
    #  [0,0,0]
    #  [0,0,0]]
    # in this representation of the image, an (x,y) point would be I[y][x]
    x_1 = int(np.floor(pt[0]))
    x_2 = int(x_1 + 1)
    y_1 = int(np.floor(pt[1]))
    y_2 = int(y_1 + 1)

    # what if x_0 = x_1 and/or y_0 = y_1 (happens when given one or two integer values for pt)
    # becomes line or triangle shape
    # we can counter act this by just adding 1 to the coordinate system to still have 4 points
    # this does not affect the result as we will only be looking at a line
    
    # pixel intensities at surrounding 4 points
    I_11 = I[y_1][x_1]
    I_12 = I[y_2][x_1]
    I_21 = I[y_1][x_2]
    I_22 = I[y_2][x_2]

    # generate matrix to find weights
    M = np.array([[1, x_1, y_1, x_1*y_1],
                [1, x_1, y_2, x_1*y_2],
                [1, x_2, y_1, x_2*y_1],
                [1, x_2, y_2, x_2*y_2]])

    # array for surrounding pixel intensities
    i = np.array([I_11, I_12, I_21, I_22])

    # Create A vector
    A = np.matmul(inv(M), i)

    # Calculate b
    result = A[0] + A[1] * pt[0] + A[2] * pt[1] + A[3] * pt[0] * pt[1]
    final_float = result[0]
    b = int(round(final_float))

    #------------------

    return b
