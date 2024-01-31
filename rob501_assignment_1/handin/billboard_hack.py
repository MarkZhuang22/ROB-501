# Billboard hack script file.
from operator import ne
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], 
                     [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], 
                        [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], 
                        [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    st_eq = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!
    billboard = Path(Iyd_pts.T)
    for x in range(np.min(bbox[0]), np.max(bbox[0])+1):
        for y in range(np.min(bbox[1]), np.max(bbox[1])+1):
            if billboard.contains_points(np.array([[x,y]])):
                point = np.array([x,y,1])
                new_point = np.matmul(H, point)
                #normalize
                new_point = new_point/new_point[-1]
                val = bilinear_interp(st_eq,new_point[:-1].reshape((2,1)))
                Ihack[y,x] = np.array([val,val,val])

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png');

    return Ihack
# billboard_hack()

# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Ist_eq = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Ist_pts, Iyd_pts)

    # Inverse of the homography for inverse warping
    H_inv = np.linalg.inv(H)

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    # Define the billboard path
    billboard = Path(Iyd_pts.T)

   # Generating x and y coordinates for the bounding Box
    x_coords = np.arange(np.min(bbox[0]), np.max(bbox[0]) + 1)
    y_coords = np.arange(np.min(bbox[1]), np.max(bbox[1]) + 1)

    # Make a copy of the original Iyd image to avoid modifying the input
    Ihack = Iyd.copy()

    # Iterate over inside coordinates and apply inverse homography (inverse warping)
    for x in x_coords:
        for y in y_coords:
            if billboard.contains_points(np.array([[x,y]])):
                point = np.array([x, y, 1])
                # Apply inverse homography transformation
                new_point = np.matmul(H_inv, point)
                # Convert it back to Cartesian coordinates
                new_point = new_point / new_point[-1]
                # Check if the new_point is within the valid range of the Soldiers' Tower image
                if 0 <= new_point[0] < Ist_eq.shape[1] and 0 <= new_point[1] < Ist_eq.shape[0]:
                    # Interpolate pixel value
                    pixel_val = bilinear_interp(Ist_eq, new_point[:-1].reshape((2, 1)))
                    Ihack[y, x] = np.array([pixel_val, pixel_val, pixel_val])

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png')

    return Ihack
