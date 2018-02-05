import tables
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
from torchvision import transforms
from tensorboard_logger import configure, log_value
import csv
import scipy.misc
import cv2
from test import get_features_vector
index = 0;

# Script to do random stuff. Don't think too much about it.
"""
im1 = cv2.imread("11-QIckmxyL.jpg")
im2 = cv2.imread("2173FIyUtQL.jpg")
im3 = cv2.imread("518IgPLpzdL._SY300_.jpg")
im4 = cv2.imread("519xzHAMXAL._SX300_.jpg")

print(get_features_vector(im1))
print(get_features_vector(im2))
print(get_features_vector(im3))
print(get_features_vector(im4))
"""
hdf5_file = tables.open_file("datasets/training_set", "r")
# To access images array:
images = hdf5_file.root.images
# keep in mind the images variable is pointing to images on hard disk. So access will be slower.
# If your memory is big enough to store all images, you can put this image array into memory by doing:
# images = np.array(file.root.images)
 # To access labels:
labels = hdf5_file.root.labels


scipy.misc.imsave("2173FIyUtQL-reco1.jpg", np.transpose(images[7093], (1,2,0)))
scipy.misc.imsave("2173FIyUtQL-reco2.jpg", np.transpose(images[4949], (1,2,0)))
scipy.misc.imsave("2173FIyUtQL-reco3.jpg", np.transpose(images[6011], (1,2,0)))
scipy.misc.imsave("2173FIyUtQL-reco4.jpg", np.transpose(images[3804], (1,2,0)))
scipy.misc.imsave("2173FIyUtQL-reco5.jpg", np.transpose(images[5600], (1,2,0)))

#scipy.misc.imsave("518IgPLpzdL._SY300_-reco1.jpg", np.transpose(images[28107], (1,2,0)))
#scipy.misc.imsave("518IgPLpzdL._SY300_-reco2.jpg", np.transpose(images[26854], (1,2,0)))
#scipy.misc.imsave("518IgPLpzdL._SY300_-reco3.jpg", np.transpose(images[31957], (1,2,0)))
#scipy.misc.imsave("518IgPLpzdL._SY300_-reco4.jpg", np.transpose(images[20002], (1,2,0)))
#scipy.misc.imsave("518IgPLpzdL._SY300_-reco5.jpg", np.transpose(images[10507], (1,2,0)))

scipy.misc.imsave("519xzHAMXAL._SX300_-reco1.jpg", np.transpose(images[409], (1,2,0)))
scipy.misc.imsave("519xzHAMXAL._SX300_-reco2.jpg", np.transpose(images[54872], (1,2,0)))
scipy.misc.imsave("519xzHAMXAL._SX300_-reco3.jpg", np.transpose(images[2923], (1,2,0)))
scipy.misc.imsave("519xzHAMXAL._SX300_-reco4.jpg", np.transpose(images[56023], (1,2,0)))
scipy.misc.imsave("519xzHAMXAL._SX300_-reco5.jpg", np.transpose(images[52370], (1,2,0)))






