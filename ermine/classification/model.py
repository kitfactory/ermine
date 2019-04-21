import os
import tensorflow as tf
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket


class SimpleCNN(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
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
        cls = OptionInfo(
            name='Class',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        return [model, size, channel, cls]

    def run(self, bucket: Bucket):
        size = int(self.options['Size'])
        channel = int(self.options['Channel'])
        cls = int(self.options['Class'])

        kernel_size = (3, 3)
        max_pool_size = (2, 2)
        print("Input Layer shape = (", size, "," ,size ,",", channel, ")" )
        input_layer = tf.keras.Input(shape=(size, size, channel, )) # batchサイズで受け取り

        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(input_layer)
        cnn = tf.keras.layers.Dropout(0.1)(cnn)
        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

        fc = tf.keras.layers.Flatten()(cnn)
        fc = tf.keras.layers.Dense(1024, activation='relu')(fc)
        softmax = tf.keras.layers.Dense(cls, activation='softmax')(fc)
        model = tf.keras.Model(inputs=input_layer, outputs=softmax)
        bucket[self.options['Model']] = model

class ResNeXt50(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        model = OptionInfo(
            name='Model',
            direction=OptionDirection.OUTPUT,
            values=['MODEL'])
        classes = OptionInfo(
            name='Class',
            direction=OptionDirection.PARAMETER,
            values=['10']
        )
        return [model,classes]

    def run(self, bucket: Bucket):
        classes = int(self.options['Class'])
        model = keras.applications.resnext.ResNeXt50(include_top=True, classes=classes)
        bucket[self.options['Model']] = model


'''
tf.keras.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs
)
'''