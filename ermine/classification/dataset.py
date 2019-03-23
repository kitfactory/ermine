from typing import List
import tensorflow as tf
import numpy as np
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket
# , LogUtil


class DirectoryClassDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        datadir = OptionInfo(
            name='DataDir', direction=OptionDirection.PARAMETER, values=['data'])
        classes = OptionInfo(
            name='Classes', direction=OptionDirection.PARAMETER, values=['2'])
        dest = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        return [datadir, classes, dest]

    def run(self, bucket: Bucket):
        pass


class CsvClassifyDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        datadir = OptionInfo(
            name='DataDir',
            direction=OptionDirection.PARAMETER,
            values=['data.csv'])
        image = OptionInfo(
            name='Image', direction=OptionDirection.PARAMETER, values=['image'])
        label = OptionInfo(
            name='Label', direction=OptionDirection.PARAMETER, values=['label'])
        classes = OptionInfo(
            name='Classes', direction=OptionDirection.PARAMETER, values=['2'])
        dest = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        return [datadir, image, label, classes, dest]

    def run(self):
        pass


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
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        bucket[self.options['TrainDataset']] = dataset
        x_test = x_test/255.0
        x_train = x_test.astype(np.float32)
        testset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        bucket[self.options['TestDataset']] = testset


class FashionMnistDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
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
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train/255.0
        x_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_train)
        y_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_train)
        bucket[self.options['TrainDataset']] = x_train_ds.zip(y_train_ds)
        x_test = x_test/255.0
        x_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_test)
        bucket[self.options['TestDataset']] = x_test_ds.zip(y_test_ds)


class DatasetPipeline(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
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
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train/255.0
        x_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_train)
        y_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_train)
        bucket[self.options['TrainDataset']] = x_train_ds.zip(y_train_ds)
        x_test = x_test/255.0
        x_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_test)
        bucket[self.options['TestDataset']] = x_test_ds.zip(y_test_ds)


