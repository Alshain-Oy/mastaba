#!/usr/bin/env python3


import sys
sys.path.append('../')

import libMastaba
import cv2
import numpy as np

# Load images to memory

clean_image = cv2.imread( sys.argv[1] ) # <- clean / golden sample ie image without particles
test_image = cv2.imread( sys.argv[2] ) # <- image to be tested

# Convert images to grayscale
bw_clean_image = cv2.cvtColor( clean_image, cv2.COLOR_BGR2GRAY )
bw_test_image = cv2.cvtColor( clean_image, cv2.COLOR_BGR2GRAY )



# align images
offset = libMastaba.FullViewAnalysis.find_image_offset( bw_clean_image, bw_test_image )

# generate shifted test image for comparison
offsetted_image = libMastaba.FullViewAnalysis.gen_offset_image( bw_test_image, *offset )

# generate exclusion mask
mask = np.zeros( offsetted_image.shape, dtype = np.uint8 )
border = offset[-1]
# exclude border area from analysis
mask[border:-border, border:-border] = 255


# Output visualisation
outimage = test_image.copy()

# Compare offsetted test image and original golden sample, excluding image border areas and discard small fluctuations in image brightness
delta, error_idx = libMastaba.FullViewAnalysis.find_errors( outimage, offsetted_image, bw_clean_image, mask, err_threshold = 25 )

# convert delta into a normal image
deltaimg = (delta*255).astype( np.uint8 )

# find errors
contours, hierarchy = cv2.findContours( deltaimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

# create statistics
errors = []
for cnt in contours:
    # Fit a minimum area rectangle over the found defect
    rect = cv2.minAreaRect( cnt )

    # Area of the minimum area rectangle
    area = rect[1][0] * rect[1][1] 

    # aspect ratio of minimum area rectangle (always <= 1)
    aspect = min(rect[1]) / max(rect[1])

    # Centre of the defect
    centre = rect[0]

    # length of the perimeter of the defect
    length = cv2.arcLength( cnt )

    entry = { "area": area, "aspect_ratio": aspect, "centre": centre, "lenght": length }
    entry.append( errors )



# print errors
import pprint
pprint.pprint( errors )





