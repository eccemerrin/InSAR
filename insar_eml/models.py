#imports
import tensorflow as tf
from insar_eml.metrics import ssim_loss

#adjustments
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_image_data_format('channels_last')


class insar_model():
    def __init__(self,
                 filter_size= 64 ,
                 kernel_size=[(2,3,3), (1,3,3)],
                 max_pooling_kernel = [(3,1,1)],
                 padding='same'):

        self.kernel_size = kernel_size
        self.max_pooling_kernel = max_pooling_kernel
        self.filter_size = filter_size
        self.padding = padding


    def __call__(self, model_input, topology_input):
        model_input = tf.keras.Input(shape=(9, 40, 40, 1))
        topology_input = tf.keras.Input(shape=(1, 40, 40, 1))

        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(model_input)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling3D(self.max_pooling_kernel[0])(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling3D(self.max_pooling_kernel[1])(x)

        combined = tf.concat([x, topology_input], axis=-1)

        y = tf.keras.layers.Conv3D(64, self.kernel_size[1], padding = self.padding)(combined)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv3D(64, self.kernel_size[1], padding = self.padding)(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv3D(64, self.kernel_size[1], padding = self.padding)(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv3D(64, self.kernel_size[1], padding = self.padding)(y)
        y = tf.math.reduce_sum(y, axis=-1)
        model_output = tf.keras.layers.Reshape((40, 40, 1))(y)

        model = tf.keras.models.Model(inputs = [model_input, topology_input], outputs = model_output, name="insar_model")
        print(type(model))
        model.compile(loss = ssim_loss, optimizer = "Adam")

        return model





