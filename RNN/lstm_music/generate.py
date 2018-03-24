# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

from utils import *
from network import *

def prepare_sequences(notes, pitch_names,num_pitch):
    sequences_length = 100

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
    normalized_input = np.reshape(network_input, (n_patterns, sequences_length, 1))

    # 将输入标准化
    normalized_input = normalized_input / float(num_pitch)

    return (network_input,normalized_input)


