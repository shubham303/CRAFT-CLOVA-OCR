import numpy as np

I = 5
debug = True
output_folder = "output"
skip_if_done=False
preprocessors=["crop", "deskew","crop"]



# Deskew operation
IMG_ERODE_KERNEL_SIZE = 5
EDGE_LOWER_THRESHOLD= 50
EDGE_UPPER_THRESHOLD=100
RHO = 1
THETA= np.pi/180
MIN_VOTE_HOUGH= 50
MIN_LINE_LENGTH= 100
MAX_LINE_GAP= 20
