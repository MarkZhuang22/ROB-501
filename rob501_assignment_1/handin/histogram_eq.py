import numpy as np
import matplotlib.pyplot as plt

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale. cool
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    image = I.flatten()
    J = np.empty(image.shape)
    nbins = 256
    rang = (0,255)
    hist, bin_edges = np.histogram(image, bins=nbins, range=rang)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    c = np.round(255*cdf).astype('uint8')

    # map f(i) = c(i)
    for i in range(0,image.shape[0]):
        J[i] = c[image[i]]
    J = J.reshape(I.shape)
    
    #------------------

    return J
