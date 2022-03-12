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


train_x, train_y, topology = utils.create_dataset()
test_x, test_y, test_topology = utils.create_dataset()

model = models.insar_model()

normalized_train_x = utils.normalize(train_x)
normalized_train_y = utils.normalize(train_y)
normalized_topology = utils.normalize(topology)

model.compile(loss = utils.ssim_loss, optimizer = "Adam")

model.fit([normalized_train_x, normalized_topology], normalized_train_y, epochs = 3)
preds_model = model.predict([test_x, test_topology])

mse = tf.keras.losses.MeanSquaredError()
print(mse(normalized_train_y.reshape(500,40,40,1), preds_model).numpy())
