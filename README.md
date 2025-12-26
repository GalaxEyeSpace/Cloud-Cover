# Cloud-Cover

Develop algos for Cloud Cover Estimation.

Approach 1:
Use thresholding method to get basic mask of clouds (this includes snow, high reflectance surfaces etc.)
Use this mask as an input to a deep learning model to get a better complete estimation.

Approach 2:
For a given image with clouds, get a reference cloudless image of the same location (Sentinel-2) and then compare them to get a estimate of cloud cover based on thresholding.