import sys
import skimage
import numpy as np
from skimage import io
from scipy import ndimage
from scipy.ndimage import uniform_filter



imgOrig = io.imread(sys.argv[1])
img1 = np.double(imgOrig)
# io.imshow(img1)
# io.show()
lMin = 7
localNeigh = np.zeros((lMin, lMin), np.uint8) 
muImg = uniform_filter(img1, lMin)
# io.imshow(muImg)
# io.show()

maskKrlDilate = np.zeros((2*lMin+1, 2*lMin+1), np.uint8)
maskKrlDilate[0, 0] = 1
maskKrlDilate[2*lMin, 2*lMin] = 1
maskKrlDilate[lMin, 0] = 1
maskKrlDilate[2*lMin, 0] = 1
maskKrlDilate[0, lMin] = 1
maskKrlDilate[0, 2*lMin] = 1
maskKrlDilate[lMin, 2*lMin] = 1
maskKrlDilate[2*lMin, lMin] = 1

temp1 = ndimage.grey_dilation(muImg, size=(lMin,lMin), structure = maskKrlDilate)
out1 = img1-temp1
out1[out1 < 0] = 0
out1 *= out1
io.imshow(out1)
io.show()
