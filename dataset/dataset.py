from types import List
import tensorflow as tf
from base.ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket, LogUtil


class DirectoryClassDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        datadir = OptionInfo(
            name='DataDir', direction=OptionDirection.OTHER, values=['data'])
        classes = OptionInfo(
            name='Classes', direction=OptionDirection.OTHER, values=['2'])
        dest = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        return [datadir, classes, dest]

    def play(self, bucket: Bucket):
        pass


class CsvDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        datadir = OptionInfo(
            name='DataDir',
            direction=OptionDirection.OTHER,
            values=['data.csv'])
        image = OptionInfo(
            name='Image', direction=OptionDirection.OTHER, values=['image'])
        label = OptionInfo(
            name='Label', direction=OptionDirection.OTHER, values=['label'])
        classes = OptionInfo(
            name='Classes', direction=OptionDirection.OTHER, values=['2'])
        dest = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        return [datadir, image, label, classes, dest]

    def play(self, bucket: Bucket):
        pass


class MnistTrainDataset(ErmineUnit):
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

    def play(self, bucket: Bucket):
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        train = x_train_dataset.zip(y_train_dataset)
        bucket[self.options['TrainDataset']] = train
        x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_test_dataset = tf.data.Dataset.from_tensor_slices(y_test)
        test = x_test_dataset.zip(y_test_dataset)
        bucket[self.options['TestDataset']] = test
        print('mnist train dataset load to {}'.format(
            self.options['TrainDataset']))
        print('mnist test dataset load to {}'.format(
            self.options['TestDataset']))


class FashionMnistTrainDataset(ErmineUnit):
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

    def play(self, bucket: Bucket):
        (x_train,
         y_train), (x_test,
                    y_test) = tf.keras.datasets.fashion_minist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        train = x_train_dataset.zip(y_train_dataset)
        bucket[self.options['TrainDataset']] = train
        x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_test_dataset = tf.data.Dataset.from_tensor_slices(y_test)
        test = x_test_dataset.zip(y_test_dataset)
        bucket[self.options['TestDataset']] = test
        LogUtil.debug('fashion mnist train dataset load to {}'.format(
            self.options['TrainDataset']))
        LogUtil.debug('fashion mnist test dataset load to {}'.format(
            self.options['TestDataset']))
