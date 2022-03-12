#imports
import tensorflow as tf

#adjustments
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_image_data_format('channels_last')


def insar_model():
    model_input = tf.keras.Input(shape=(9, 40, 40, 1))
    topology_input = tf.keras.Input(shape=(1, 40, 40, 1))

    x = tf.keras.layers.Conv3D(64, (2,3,3), (1, 1, 1), padding = "same")(model_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(64, (2,3,3), (1, 1, 1), padding = "same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(64, (2,3,3), (1, 1, 1), padding = "same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D((3,1,1))(x)
    x = tf.keras.layers.Conv3D(64, (2,3,3), (1, 1, 1), padding = "same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D((3,1,1))(x)

    combined = tf.concat([x, topology_input], axis=-1)

    y = tf.keras.layers.Conv3D(64, (1,3,3), padding = "same")(combined)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1,3,3),  padding = "same")(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1,3,3), padding = "same")(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1,3,3), padding = "same")(y)
    y = tf.math.reduce_sum(y, axis=-1)
    model_output = tf.keras.layers.Reshape((40, 40, 1))(y)

    model = tf.keras.models.Model(inputs = [model_input, topology_input], outputs = model_output, name="insar_model")
    model.summary()
    return model




