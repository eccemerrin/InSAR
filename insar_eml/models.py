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


def create_vae_model():

    class Sampling(tf.keras.layers.Layer):
         #Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
        def call(self, inputs):
             z_mean, z_log_var = inputs
             batch = tf.shape(z_mean)[0]
             dim = tf.shape(z_mean)[1]
             epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
             return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    latent_dim = 2
    encoder_inputs = tf.keras.Input(shape = (9, 40, 40, 1))

    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")(encoder_inputs)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")(x)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")(x)
    x = tf.keras.layers.MaxPooling3D((3, 1, 1))(x)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding = 'same', activation="relu")(x)
    x = tf.keras.layers.MaxPooling3D((3, 1, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = tf.keras.Model(encoder_inputs, z, name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    y = tf.keras.layers.Dense(40 * 40 * 64, activation="relu")(latent_inputs)
    y = tf.keras.layers.Reshape((1, 40, 40, 64))(y)
    y = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3),
                                         strides=(1, 1, 1), activation="relu", padding="same")(y)
    y = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3),
                                         strides=(1, 1, 1), activation="relu", padding="same")(y)
    y = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(1, 3, 3),
                                         strides=(1, 1, 1), activation="sigmoid", padding="same")(y)
    decoder_outputs = tf.math.reduce_sum(y, axis = -1)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    outputs = decoder(z)
    vae = tf.keras.Model(inputs=encoder_inputs, outputs=outputs, name="vae")

    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)

    # Loss and optimizer.
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Configure the model for training.
    vae.compile(optimizer, loss=loss_fn)

    return vae


def volcanic_encoder_decoder():
    model_input = tf.keras.Input(shape=(9, 40, 40, 1))
    second_input = tf.keras.Input(shape=(1, 40, 40, 1))

    # encoder
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(model_input)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_1 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_1)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(1, 1, 1))(x)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_2 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_2)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(1, 1, 1))(x)

    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x_concat_3 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x_concat_3)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), strides=(1, 1, 1))(x)

    # bottleneck
    x = tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(512, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    # decoder
    x = tf.keras.layers.Conv3DTranspose(256, (2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    concat_3 = tf.concat([x, x_concat_3], axis=-1)
    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(concat_3)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    x = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    concat_2 = tf.concat([x, x_concat_2], axis=-1)
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(concat_2)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    x = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    concat_1 = tf.concat([x, x_concat_1], axis=-1)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(concat_1)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    # beginning of our architecture:
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D((3, 1, 1))(x)
    x = tf.keras.layers.Conv3D(64, (2, 3, 3), (1, 1, 1), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D((3, 1, 1))(x)

    combined = tf.concat([x, second_input], axis=-1)

    y = tf.keras.layers.Conv3D(64, (1, 3, 3), padding="same")(combined)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1, 3, 3), padding="same")(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1, 3, 3), padding="same")(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Conv3D(64, (1, 3, 3), padding="same")(y)
    y = tf.math.reduce_sum(y, axis=-1)
    model_output = tf.keras.layers.Reshape((40, 40, 1))(y)

    model = tf.keras.Model(inputs=[model_input, second_input], outputs=model_output,
                                  name="volcanic_enocder_decoder")
    model.summary()

    return model