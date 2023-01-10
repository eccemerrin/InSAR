import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import numpy as np
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_image_data_format('channels_last')