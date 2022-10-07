# mastaba
Mastaba Machine Vision Library


## Usage
```python

# Parameters

# Nomenclature
# Fiducial, template = pattern used for matching
# Level = level in the image pyramid, size of the image is 2^(-level) compared to the original


### SEARCH RELATED PARAMETERS

# searchLevel = integer, 0..N
# at what level the first search for the fiducial is performed
#    - higher level -> smaller image and faster initial search, but possibly more false matches
#    - if too high, small patterns cannot be distinguished
ncc.configure( template, searchLevel = 3 )  

# refineLevel = integer, 0..N
# at what level the last search for the fiducial is performed
#    - refineLevel <= searchLevel
ncc.configure( template, refineLevel = 0 )  

# searchCorrelation = float, 0..1
# Minimum correlation for the first search stage
# Pattern is matched in different rotations using the provided mask to block unwanted parts
# Can produce lot of false positives at lower correlation levels
ncc.configure( template, searchCorrelation = 0.9 )

# searchCorrelationVerify = float, 0..1
# Minimum correlation to verify each search results
# Uses a much stringent algorithm to check if each result from search stage is good enough to be refined further
# Should be smaller or same as refineCorrelation
ncc.configure( template, searchCorrelationVerify = 0.8 )

# refineCorrelation = float, 0..1
# Minimum correlation to accept a pattern
# All results returned have correlation better that this
ncc.configure( template, refineCorrelation = 0.85 )

# nmsRadius = integer, 1..N
# Minimum distance in pixels (in the original image scale ie, leve 0) between two patterns
# All patterns within nmsRadius are grouped and non-maxmimal suppression (nms) is used to select the best result for further refinement
ncc.configure( template, nmsRadius = 32 )

# extraPadding = boolean, true / false
# Is the search image padded so that patterns can be found in the edge regions of the image
# As the search image is larger, the search takes longer 
ncc.configure( template, extraPadding = True )

# useTemplatePadding = boolean, true / false
# Is the template padded so that rotations do not clip corners of the template when matched
# As the template is larger, the search takes longer
ncc.configure( template, useTemplatePadding = True )


# useMaskForMatching = boolean, true / false
# Is the mask used for matching ie. is the mask used to mask out parts of the template
ncc.configure( template, useMaskForMatching = True )

# corrMethod = cv2.TM_CCOEFF_NORMED or cv2.TM_CCORR_NORMED
# Which OpenCV correlation method is used for pattern search
# cv2.TM_CCORR_NORMED can be used easily with mask but is less discriminating
# cv2.TM_CCOEFF_NORMED is far more stringent when matching, but when used in conjuncton with mask, it can be more unstable
ncc.configure( template, corrMethod = cv2.TM_CCORR_NORMED )

# ROI = [x0, x0, width, height]
# Region of interest used for seaching, used to select only small portion of the image as the search area
# Can speed up the search significantly as search time is proportional to image area
ncc.configure( template, ROI = [0,0, image_width, image_height] )

# drawResults = boolean, true / false
# Are results drawn (as arrows showing orientation) in the returned image
ncc.configure( template, drawResults = True )

# drawROI = boolean, true / false
# Is the ROI used drawn in the returned image
ncc.configure( template, drawROI = True )

# scoreOrdering = boolean, true / false
# Are results sorted by correlation, best first
ncc.configure( template, scoreOrdering = True )

# radialOrdering = boolean, true / false
# Are results sorted by distance from image centre, closest first
# If set to True, it will override scoreOrdering
ncc.configure( template, radialOrdering = False )

# fromCentre = boolean, true / false
# Are result coordinates given referenced to image centre
ncc.configure( template, fromCentre = False )

# debug = boolean, true / false
# Debug mode on/off, when on, information about the search progress is shown
ncc.configure( template, debug = False )


```
