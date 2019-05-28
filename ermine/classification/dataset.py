from typing import List
import os
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket
from abc import abstractmethod
# , LogUtil

class ImageClassificationDataset(ErmineUnit):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        train = OptionInfo(
            name='TrainDataset',
            direction=OptionDirection.OUTPUT,
            values=['TRAIN_DATASET'])

        train_size = OptionInfo(
            name='TrainDatasetSize',
            direction=OptionDirection.OUTPUT,
            values=['TRAIN_DATASET_SIZE'])
        
        validation = OptionInfo(
            name='ValidationDataset',
            direction=OptionDirection.OUTPUT,
            values=['VALIDATION_DATASET'])

        validation_size = OptionInfo(
            name='ValidationDatasetSize',
            direction=OptionDirection.OUTPUT,
            values=['VALIDATION_DATASET_SIZE'])

        test = OptionInfo(
            name='TestDataset',
            direction=OptionDirection.OUTPUT,
            values=['TEST_DATASET'])

        test_size = OptionInfo(
            name='TestDatasetSize',
            direction=OptionDirection.OUTPUT,
            values=['TEST_DATASET_SIZE'])

        return [train, train_size, validation, validation_size, test, test_size]

    @abstractmethod
    def load_dataset(self)->((tf.data.Dataset,int),(tf.data.Dataset,int),(tf.data.Dataset,int)):
        pass
    
    def run(self, bucket: Bucket):
        (train,train_size),(validation,validation_size),(test, test_size) = self.load_dataset()
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['TrainDatasetSize']] = train_size
        bucket[self.options['ValidationDataset']] = validation
        bucket[self.options['ValidationDatasetSize']] = validation_size
        bucket[self.options['TestDataset']] = test
        bucket[self.options['TestDatasetSize']] = test_size
        print(bucket)

'''
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


'''


class CsvClassifyDataset(ImageClassificationDataset):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:

        infos = ImageClassificationDataset.prepare_option_infos()

        train = OptionInfo(
            name='TrainFile',
            direction=OptionDirection.PARAMETER,
            values=['train.csv'])
        test = OptionInfo(
            name='TestFile',
            direction=OptionDirection.PARAMETER,
            values=['test.csv'])
        image = OptionInfo(
            name='Image', direction=OptionDirection.PARAMETER, values=['image'])
        label = OptionInfo(
            name='Label', direction=OptionDirection.PARAMETER, values=['label'])
        return infos + [train, test, image, label]


    def load_dataset(self)->((tf.data.Dataset,int),(tf.data.Dataset,int),(tf.data.Dataset,int)):
        train_csv = pd.read_csv(self.options['TrainFile'])
        # shuffle
        train_csv = train_csv.sample(frac=1).reset_index(drop=True)
        labels = train_csv[self.options['Label']].values.astype(np.uint8)
        max_label = labels.max() + 1
        labels = tf.keras.utils.to_categorical(labels,num_classes=max_label)
        train_data_num = len(labels)
        images = train_csv[self.options['Image']].values

        trains = (train_data_num // 10 * 9) + (train_data_num % 10)
        validation = (train_data_num // 10)
        classes = max_label
        test_csv = pd.read_csv(self.options['TestFile'])
        test_labels = test_csv[self.options['Label']].values.astype(np.uint8)
        test_labels = tf.keras.utils.to_categorical(test_labels,num_classes=max_label)
        test_images = test_csv[self.options['Image']].values
        test_data_num = len(test_labels)
 
        def map_fn(x,y):
            raw = tf.io.read_file(x)
            img = tf.io.decode_image(raw, channels=3)
            img = tf.cast(img, dtype=tf.float32)
            img = img/255.0
            return (img,y)

        train_ds = tf.data.Dataset.from_tensor_slices((images[:trains],labels[:trains])).map(map_func=map_fn).repeat()
        validation_ds = tf.data.Dataset.from_tensor_slices((images[trains:],labels[trains:])).map(map_func=map_fn).repeat()
        test_ds = tf.data.Dataset.from_tensor_slices((test_images[trains:],test_labels[trains:])).map(map_func=map_fn).repeat()

        return ((train_ds,trains),(validation_ds,validation),(test_ds,test_data_num))

class MnistDataset(ImageClassificationDataset):
    def __init__(self):
        super().__init__()

    def load_dataset(self)->((tf.data.Dataset,int),(tf.data.Dataset,int),(tf.data.Dataset,int)):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train/255.0
        x_train = x_train.astype(np.float32)
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        y_train = tf.keras.utils.to_categorical(y_train,10)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        dataset = dataset.shuffle(60000, seed=10)
        train = dataset.take(54000)
        validation = dataset.skip(54000).take(6000)
       
        x_test = x_test/255.0
        x_test = x_test.astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0],28,28, 1)

        y_test = tf.keras.utils.to_categorical(y_test,10)
        testset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        testset = testset.repeat()
        return ((train,54000),(validation,6000),(testset,10000))


class FashionMnistDataset(ErmineUnit):
    def __init__(self):
        super().__init__()

    def load_dataset(self)->((tf.data.Dataset,int),(tf.data.Dataset,int),(tf.data.Dataset,int)):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train/255.0
        x_train = x_train.astype(np.float32)
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        y_train = tf.keras.utils.to_categorical(y_train,10)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        dataset = dataset.shuffle(60000, seed=10)
        train = dataset.take(54000).repeat()
        validation = dataset.skip(54000).take(6000).repeat()
       
        x_test = x_test/255.0
        x_test = x_test.astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0],28,28, 1)
        y_test = tf.keras.utils.to_categorical(y_test,10)
        testset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        testset = testset.repeat()
        return ((train,54000),(validation,6000),(testset,10000))


'''
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


'''