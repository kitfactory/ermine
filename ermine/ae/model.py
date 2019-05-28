import tensorflow as tf
import os
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket
from abc import abstractmethod 

class SimpleAutoEncoder(ErmineUnit):

    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(cls):
        model = OptionInfo(
            name='Model',
            direction=OptionDirection.OUTPUT,
            values=['MODEL'])
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['28'])
        channel = OptionInfo(
            name='Channel',
            direction=OptionDirection.PARAMETER,
            values=['1'])
        return [model, size, channel]

    def run(self, bucket:Bucket):
        size = int(self.options['Size'])
        channel = int(self.options['Channel'])
        model_dest = self.options['Model']

        print('size', size, 'channel', channel)
        inputs = tf.keras.layers.Input(shape=(size,size,channel))
        input2 = tf.keras.layers.Reshape((size*size*channel,))(inputs)

        mid1 = tf.keras.layers.Dense(size*size*channel//4, activation='relu')(input2)
        mid2 = tf.keras.layers.Dense(size*size*channel//16, activation='relu')(mid1)
        mid3 = tf.keras.layers.Dense(size*size*channel//64, activation='relu')(mid2)
        mid4 = tf.keras.layers.Dense(size*size*channel//64, activation='relu')(mid3)
        mid5 = tf.keras.layers.Dense(size*size*channel//16, activation='relu')(mid4)
        mid6 = tf.keras.layers.Dense(size*size*channel//4, activation='relu')(mid5)
        last = tf.keras.layers.Dense(size*size*channel, activation='sigmoid')(mid6)  # use sigmoid activation at the last layer.
        
        reshape = tf.keras.layers.Reshape((size, size, channel))(last)
        model = tf.keras.Model(inputs=inputs, outputs=reshape)
        bucket[model_dest] = model
