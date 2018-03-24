# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

from utils import *
from network import *

def train():
    notes = get_notes()
    
    num_pitch = len(set(notes))

    network_input,network_output = prepare_sequences(notes,num_pitch)

    model = network_model(network_input,num_pitch)

    filepath = 'weights-{epoch:02d}-{loss:.4f}.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode="min" 
    )

    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=100, batch_size=32, callbacks=callbacks_list)



def prepare_sequences(notes, num_pitch):
    sequences_length = 100

    pitch_names = sorted(set(item for item in notes))

    pitch_to_int = dict((pitch,num) for num,pitch in enumerate(pitch_names))

    network_input = []
    network_output = []

    for i in range(0,len(notes) - sequences_length,1):
        sequences_in = notes[i:i + sequences_length]
        sequences_out = notes[i + sequences_length]

        network_input.append([pitch_to_int[char] for char in sequences_in])
        network_output.append([pitch_to_int[sequences_out]])

    
    n_patterns = len(network_input)

    # 将输入的形状转换成 LSTM 模型可以接受的
    network_input = np.reshape(network_input, (n_patterns, sequences_length, 1))

    # 将输入标准化
    network_input = network_input / float(num_pitch)



    #network_output = tf.keras.utils.to_categorical(network_output)
    network_output = tf.keras.utils.to_categorical(network_output)

    return (network_input,network_output)



if __name__ == "__main__":
    train()
    



