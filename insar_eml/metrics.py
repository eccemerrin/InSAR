#imports
import numpy as np
import tensorflow as tf

def normalize(data):
    normalized_data = data / np.max(np.abs(data))
    normalized_data += 1
    return normalized_data

# Loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))