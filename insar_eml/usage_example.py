#imports
import os
import sys
import inspect
import tensorflow as tf

#to solve relative import issues
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from insar_eml import models
from insar_eml import utils

#please indicate the number of pixels. It will allow you to change input dim. of insar_model and input data.
num_pixels = 48

train_x, train_y, topology = utils.create_dataset( num_pixels = num_pixels)
test_x, test_y, test_topology = utils.create_dataset(num_pixels = num_pixels)

#uncomment the line of the model you want to try
model = models.parametarized_insar_model(num_pixels = num_pixels)
#model = models.create_vea_model()
#model = models.volcanic_encoder_decoder()

normalized_train_x = utils.normalize(train_x)
normalized_train_y = utils.normalize(train_y)
normalized_topology = utils.normalize(topology)

#this loss is for insar_model model. If you want to try another model please choose mse for both volcanic encoder decoder
# model and vae model. Choose sgd optimizer for volcanic encoder decoder and choose adam for vae.
model.compile(loss = utils.ssim_loss, optimizer = "Adam")

model.fit([normalized_train_x, normalized_topology], normalized_train_y, epochs = 3)
preds_model = model.predict([test_x, test_topology])

mse = tf.keras.losses.MeanSquaredError()
print(mse(normalized_train_y.reshape(500,num_pixels, num_pixels, 1), preds_model).numpy())
