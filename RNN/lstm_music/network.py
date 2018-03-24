# -*- coding: UTF-8 -*-

import tensorflow as tf

def network_model(inputs, num_pitch, weights_file=None):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(
        512,
        input_shape=(inputs.shape[1],inputs.shape[2]),
        return_sequences=True
    ))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(
        512,
        return_sequences=True
    ))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(
        512,
        return_sequences=False
    ))

    model.add(tf.keras.layers.Dense(256))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(num_pitch))

    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file is not None:
        # HDF5 Load Weights
        model.load_weights(weights_file)

    return model
