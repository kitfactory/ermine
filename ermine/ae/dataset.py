from typing import List
import os
import tensorflow as tf
import numpy as np
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket

class MnistDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    def prepare_option_infos(self) -> List[OptionInfo]:
        train = OptionInfo(
            name='TrainDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        test = OptionInfo(
            name='TestDataset',
            direction=OptionDirection.OUTPUT,
            values=['TEST_DATASET'])
        return [train, test]

    def run(self, bucket: Bucket):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train/255.0
        x_train = x_train.astype(np.float32)
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
        bucket[self.options['TrainDataset']] = dataset
        x_test = x_test/255.0
        x_test = x_test.astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0],28,28, 1)
        testset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
        bucket[self.options['TestDataset']] = testset
