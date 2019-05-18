from typing import List
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket
# , LogUtil



class DirectoryClassDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        datadir = OptionInfo(
            name='DataDir',
            direction=OptionDirection.PARAMETER,
            values=['data'])        
        classes = OptionInfo(
            name='Classes',
            direction=OptionDirection.PARAMETER,
            values=['2'])
        ext = OptionInfo(
            name='Ext',
            direction=OptionDirection.PARAMETER,
            values=['.png','.jpg','.bmp'])
        dest = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.OUTPUT,
            values=['DATASET'])
        stats = OptionInfo(
            name='Stats',
            direction=OptionDirection.OUTPUT,
            values=['True', 'False'])
        return [datadir, classes, ext, dest, stats]

    
    def __print_stats(self, datasets:List[tf.data.Dataset]):
        print("print stats!!")
        total = 0
        for idx, d in enumerate(datasets):
            iterator = d.make_one_shot_iterator()
            next_item = iterator.get_next()
            counter = 0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    while True:
                        sess.run(next_item)
                        counter = counter + 1
                        total = total + 1
                except tf.errors.OutOfRangeError as err:
                    print('label' , idx , '  counts=', counter )
        print('total', total)


    def run(self, bucket: Bucket):
        data_dir = self.options['DataDir']
        ext = self.options['Ext']

        subdirs = os.listdir(data_dir)
        datasets = []

        for idx, sub in enumerate(subdirs):
            pattern = data_dir + os.path.sep + sub + os.path.sep + "*" + ext
            print(pattern)

            def add_label(filepath):
                return (filepath, idx)

            dataset = tf.data.Dataset.list_files(pattern).map(add_label)
            
            def load_img(filepath, label):
                raw = tf.read_file(filepath)
                return (raw, label)

            dataset = dataset.map(load_img)

            # def decode_png(raw, label):
            #     img = tf.image.decode_png(raw)
            #     return (img, label)

            def decode_img(raw, label):
                img = tf.image.decode_image(raw, channels=3)
                return (img, label)

            # TODO. 画像種類とデコーダはまだ確定してない
            if ext == ".png":
                dataset = dataset.map(decode_img)
            else:
                dataset = dataset.map(decode_img)
            datasets.append(dataset)

        if bool(self.options['Stats']) == True:
            self.__print_stats(datasets)

        all_dataset: tf.data.Dataset = None
        for d in datasets:
            if all_dataset == None:
                all_dataset = d
            else:
                all_dataset = all_dataset.concatenate( d )
        bucket[self.options['DestDataset']] = all_dataset


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
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        bucket[self.options['TrainDataset']] = dataset
        x_test = x_test/255.0
        x_test = x_test.astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0],28,28, 1)
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



class Cifar10Dataset(ErmineUnit):
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
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train/255.0
        x_train = x_train.astype(np.float32)
        print("x_train.shape =" , x_train.shape)
        print("y_train.shape =" , y_train.shape)
        x_train = x_train.reshape(x_train.shape[0],32,32,3)
        y_train = y_train.reshape(y_train.shape[0])
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        bucket[self.options['TrainDataset']] = dataset
        x_test = x_test/255.0
        x_test = x_test.astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0],32,32, 3)
        y_test = y_test.reshape(y_test.shape[0])
        testset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        bucket[self.options['TestDataset']] = testset


class TFDSDataset(ErmineUnit):

    dataset_name = 'mnist'

    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(cls) -> List[OptionInfo]:
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
        # datadir = self.options['DataDir']
        datadir = self.globals['HOMEDIR'] + os.path.sep + 'dataset' + os.path.sep + TFDSDataset.dataset_name
        if os.path.exists(datadir) is False:
            print('makedir' , datadir)
            os.makedirs(datadir)
        else:
            print('dataset dir exists' , datadir)
        
        # cfg = tfds.core.BuilderConfig(name=TFDSDataset.dataset_name)
        builder = tfds.builder(name=TFDSDataset.dataset_name, data_dir=datadir)
        builder.download_and_prepare()
        train = builder.as_dataset(split=tfds.Split.TRAIN)
        test = builder.as_dataset(split=tfds.Split.TEST)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['TestDataset']] = test



class Imagenet2012Dataset(TFDSDataset):

    dataset_name = 'imagenet2012'

    def run(self, bucket: Bucket):
        # datadir = self.options['DataDir']
        datadir = self.globals['HOMEDIR'] + os.path.sep + 'dataset' + os.path.sep + Imagenet2012Dataset.dataset_name
        if os.path.exists(datadir) is False:
            print('makedir' , datadir)
            os.makedirs(datadir + os.path.sep + "downloads" + os.path.sep + "manual" + os.path.sep + "imagenet2012")
        else:
            print('dataset dir exists' , datadir)
        
        # cfg = tfds.core.BuilderConfig(name=TFDSDataset.dataset_name)
        builder = tfds.builder(name=Imagenet2012Dataset.dataset_name, data_dir=datadir)
        builder.download_and_prepare()
        train = builder.as_dataset(split=tfds.Split.TRAIN)
        test = builder.as_dataset(split=tfds.Split.TEST)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['TestDataset']] = test



class CatsVsDogDataset(TFDSDataset):

    dataset_name = 'cats_vs_dogs'

    def run(self, bucket: Bucket):
        # datadir = self.options['DataDir']
        datadir = self.globals['HOMEDIR'] + os.path.sep + 'dataset' + os.path.sep + CatsVsDogDataset.dataset_name
        if os.path.exists(datadir) is False:
            print('makedir' , datadir)
            os.makedirs(datadir + os.path.sep + "downloads" + os.path.sep + "manual" + os.path.sep + "horses_or_humans")
        else:
            print('dataset dir exists' , datadir)
        
        # cfg = tfds.core.BuilderConfig(name=TFDSDataset.dataset_name)
        builder = tfds.builder(name=CatsVsDogDataset.dataset_name, data_dir=datadir)
        builder.download_and_prepare()
        train = builder.as_dataset(split=tfds.Split.TRAIN)
        test = builder.as_dataset(split=tfds.Split.TEST)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['TestDataset']] = test


class RockPaperScissorsDataset(TFDSDataset):

    dataset_name = 'rock_paper_scissors'

    def run(self, bucket: Bucket):
        # datadir = self.options['DataDir']
        datadir = self.globals['HOMEDIR'] + os.path.sep + 'dataset' + os.path.sep + RockPaperScissorsDataset.dataset_name
        if os.path.exists(datadir) is False:
            print('makedir' , datadir)
            os.makedirs(datadir + os.path.sep + "downloads" + os.path.sep + "manual" + os.path.sep + "rock_paper_scissors")
        else:
            print('dataset dir exists' , datadir)
        
        # cfg = tfds.core.BuilderConfig(name=TFDSDataset.dataset_name)
        builder = tfds.builder(name=RockPaperScissorsDataset.dataset_name, data_dir=datadir)
        builder.download_and_prepare()
        train = builder.as_dataset(split=tfds.Split.TRAIN)
        test = builder.as_dataset(split=tfds.Split.TEST)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['TestDataset']] = test



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
        x_train = np.reshape(x_train, [28,28,1])
        x_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_train)
        y_train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_train)
        bucket[self.options['TrainDataset']] = x_train_ds.zip(y_train_ds)
        x_test = x_test/255.0
        x_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_test_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(y_test)
        bucket[self.options['TestDataset']] = x_test_ds.zip(y_test_ds)


