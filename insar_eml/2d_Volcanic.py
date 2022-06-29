import os
import sys
import inspect

#imports
import tensorflow as tf

#adjustments
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_image_data_format('channels_last')

#to solve relative import issues
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

def Volcanic_2D():
    model_input = tf.keras.Input(shape = (40, 40, 1))
    #encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same')(model_input)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_1 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_1)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_2 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_2)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_3 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_3)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_4 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x_concat_4)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides = (2,2))(x)

    #bottleneck
    x = tf.keras.layers.Conv2D(512, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    #decoder
    x = tf.keras.layers.Conv2DTranspose(256, (3,3), (2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    concat_4 = tf.concat([x, x_concat_4], axis = -1)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(concat_4)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    x = tf.keras.layers.Conv2DTranspose(128, (3,3), (2, 2))(x)
    x = tf.keras.layers.Conv2D(128,(3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x)
    concat_3 = tf.concat([x, x_concat_3], axis = -1)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding = 'same')(concat_3)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3,3), (2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x)
    concat_2 = tf.concat([x, x_concat_2], axis = -1)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same')(concat_2)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3,3), (2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout( 0.8)(x)
    concat_1 = tf.concat([x, x_concat_1], axis = -1)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same')(concat_1)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    model_output = tf.keras.layers.Dropout(0.8)(x)


    model = tf.keras.Model(inputs = model_input, outputs = model_output, name = "model")
    model.summary()
    return model


model = Volcanic_2D()