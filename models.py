#imports
import tensorflow as tf

#adjustments
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_image_data_format('channels_last')


class inSAR_model():
    def __init__(self,
                 filter_size= 64 ,
                 kernel_size=[(2,3,3), (1,3,3)],
                 max_pooling_kernel = [(3,1,1)],
                 padding='same',
                 activationf='relu',
                 ):
        self.kernel_size = kernel_size
        self.max_pooling_kernel = max_pooling_kernel
        self.filter_size = filter_size
        self.padding = padding
        self.activationf = activationf

    def __call__(self, model_input, topology_input):
        model_input = tf.keras.Input(shape=(9, 40, 40, 1))
        topology_input = tf.keras.Input(shape=(1, 40, 40, 1))

        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding )(model_input)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling3D(self.max_pooling_kernel[0])(x)
        x = tf.keras.layers.Conv3D(self.filter_size, self.kernel_size[0], (1, 1, 1), padding = self.padding)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling3D(self.max_pooling_kernel[-1])(x)

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

        insar_model = tf.keras.Model(inputs = [model_input, topology_input], outputs = model_output,
                                    name="insar_model")
        insar_model.summary()
        return insar_model
    
## Beginning of the Variational Autoencoder
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class Encoder(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.latent_dim = 2
        self.sampling = Sampling()
        self.layer1 = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1),
                                             padding = 'same', activation="relu", input_shape=(9, 40, 40, 1))
        self.layer2 = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")
        self.layer3 = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")
        self.layer4 = tf.keras.layers.MaxPooling3D((3, 1, 1))
        self.layer5 = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")
        self.layer6 = tf.keras.layers.MaxPooling3D((3, 1, 1))
        self.layer7 = tf.keras.layers.Flatten()
        self.layer8 = tf.keras.layers.Dense(16, activation="relu")
        
        self.dense_mean = tf.keras.layers.Dense(self.latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(self.latent_dim)
        
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    
class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.latent_dim = 2
        self.layer9 = tf.keras.layers.Dense(40 * 40 * 64, activation="relu", input_shape=(2,))
        self.layer10 = tf.keras.layers.Reshape((1, 40, 40, 64))
        self.layer11 = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3), 
                                                       strides=(1, 1, 1), activation="relu", padding="same")
        self.layer12 = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3), 
                                                       strides=(1, 1, 1), activation="relu", padding="same")
        self.layer13 = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3), 
                                                       strides=(1, 1, 1), activation="sigmoid", padding="same")
        
    def call(self, inputs):
        x = self.layer9(inputs)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        return tf.math.reduce_sum(x, axis = -1)
    
class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
