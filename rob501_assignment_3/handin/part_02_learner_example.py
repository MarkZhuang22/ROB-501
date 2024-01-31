import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from stereo_disparity_score import stereo_disparity_score
from stereo_disparity_best import stereo_disparity_best

# Load the stereo images and ground truth.
Il = imread("/Users/markzhuang/Desktop/rob501_fall_2023_assignment_03/images/cones/cones_image_02.png", mode='F')
Ir = imread("/Users/markzhuang/Desktop/rob501_fall_2023_assignment_03/images/cones/cones_image_06.png", mode='F')

# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
It = imread("/Users/markzhuang/Desktop/rob501_fall_2023_assignment_03/images/cones/cones_disp_02.png",  mode='F')/4.0

# Load the appropriate bounding box.
bbox = np.load("/Users/markzhuang/Desktop/rob501_fall_2023_assignment_03/data/cones_02_bounds.npy")

Id = stereo_disparity_best(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))
plt.imshow(Id, cmap = "gray")
plt.show()

