from typing import List, Dict
from enum import Enum
from abc import ABCMeta, abstractmethod

import json
import tensorflow as tf

import logging


class LogLevel(Enum):
    DEBUG: int = logging.DEBUG
    INFO: int = logging.INFO
    WARNING: int = logging.WARNING
    ERROR: int = logging.ERROR


class LogUtil():
    @staticmethod
    def init(name: str = __name__, level: int = LogLevel.DEBUG):
        LogUtil.logger = logging.getLogger(name)
        LogUtil.logger.setLevel(level)

    @staticmethod
    def debug(s: str):
        LogUtil.logger.debug(s)

    @staticmethod
    def info(s: str):
        LogUtil.logger.info(s)

    @staticmethod
    def warn(s: str):
        LogUtil.logger.warn(s)

    @staticmethod
    def error(s: str):
        LogUtil.logger.error(s)


class OptionDirection(Enum):
    INPUT = 1
    OUTPUT = 2
    OTHER = 3


class KitPlayException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


'''
{
    name: 'hoge',
    default: ['x']
    type: 1,
    direction: 1
}
'''


class OptionInfo():
    def __init__(self,
                 json: str = None,
                 name: str = None,
                 direction: OptionDirection = OptionDirection.OTHER,
                 values: List[str] = ['']):
        if json is not None:
            dict: Dict = json.load(json)
            self.name = dict['name']
            direction: Dict[int, OptionDirection] = {
                1: OptionDirection.INPUT,
                2: OptionDirection.OUTPUT,
                3: OptionDirection.OTHER
            }
            self.direction = direction[dict['direction']]
            self.values = dict['default']

        else:
            self.name = name
            self.direction = direction
            self.values = values

    def get_name(self) -> str:
        return self.name

    def get_values(self) -> List[str]:
        return self.values

    def get_direction(self) -> OptionDirection:
        return self.direction

    def get_description(self) -> str:
        return ''

    def to_string(self) -> str:
        return ''


'''
    def __init__(self, name:str, direction:OptionDirection, values:List[str] ):
        super.__init__()
        self.name = name
        self.direction = direction
        self.values = values


class OptionBuilder():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_options(self, json:str)->List[KitPart]:
        return []

'''


class Bucket(Dict):
    def __init__(self):
        super().__init__()


class KitPart(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.option_infos: List[OptionInfo] = self.prepare_option_infos()
        self.options: Dict[str, str] = self.__prepare_options_with_default()

    def get_module_name(self) -> str:
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        pass

    def get_option_infos(self) -> List[OptionInfo]:
        return self.option_infos

    def __prepare_options_with_default(self) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for oi in self.option_infos:
            option_info: OptionInfo = oi
            name = option_info.get_name()
            value = option_info.get_values()[0]
            d[name] = value
        return d

    def set_options(self, options: Dict[str, str]):
        for k in options:
            self.options[k] = options[k]

    @abstractmethod
    def play(self, bucket: Bucket):
        pass


class KitPlayer():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks: List[KitPart] = []

    def set_playlist(self, json_str: str = None, json_file: str = None):
        try:
            if json_str is None and json_file is None:
                raise KitPlayException(
                    'No specified playlist. KitPlayer needs a playlist to play.'
                )

            if json_str is not None:
                config = json.load(json_str)

            if json_file is not None:
                with tf.io.gfile.GFile(json_file, 'r') as f:
                    config = json.load(f, encoding='utf-8')

            play_list = config['playlist']

            for block_config in play_list:
                block_class = globals()[block_config['name']]
                instance = block_class()
                instance.set_options(block_config['options'])
                self.blocks.append(instance)

        except Exception as exception:
            raise KitPlayException(exception)

    def play(self, bucket: Bucket = None) -> Bucket:
        if bucket is None:
            bucket = Bucket()
        try:
            for b in self.blocks:
                b.play(bucket)
        except Exception as exception:
            raise KitPlayException(exception)

        return bucket


class DatasetLabelStats(KitPart):
    def __init__(self):
        super().__init__()


class DatasetImageStats(KitPart):
    def __init__(self):
        super().__init__()


class WebService():
    def __load_parts_list(self):
        self.parts_list = ['FooPart', 'BarPart', '']

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__load_parts_list()

    def get_module_name_list(self) -> List[str]:
        return self.parts_list


'''
    def get_options(self)->List:
        return []

    def get_defalut_value(key:str)->str:
        return ''
    
    def get_values_list(key:str)->List:
        return []

    def set_option(key:str, val:str):
        pass
'''


class FooPart(KitPart):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        o = OptionInfo(
            name='message_key',
            direction=OptionDirection.OUTPUT,
            values=['message'])
        return [o]

    def play(self, bucket: Bucket):
        k = self.options['message_key']
        bucket[k] = 'HelloWorld'
        LogUtil.debug('Set hello message to {}'.format(k))


class BarPart(KitPart):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        o = OptionInfo(
            name='message_key', direction=OptionDirection.INPUT, values=[''])
        return [o]

    def play(self, bucket: Bucket):
        k = self.options['message_key']
        LogUtil.debug('Get message is {}'.format(bucket[k]))


class JsonInitialParameter(KitPart):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        o = OptionInfo(
            name='json_path',
            direction=OptionDirection.INPUT,
            values=['param.json'])
        return [o]

    def play(self, bucket: Bucket):
        json_file: str = self.options['json_path']
        if json_file is not None:
            with tf.io.gfile.GFile(json_file, 'r') as f:
                d = json.load(f, encoding='utf-8')


class DirectoryClassDataset(KitPart):
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


class CsvDataset(KitPart):
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


class MnistTrainDataset(KitPart):
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


class FashionMnistTrainDataset(KitPart):
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


class ShuffleDataset(KitPart):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        dataset = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.INPUT,
            values=['DATASET'])
        size = OptionInfo(
            name='ShuffleSize',
            direction=OptionDirection.OTHER,
            values=['100000'])
        seed = OptionInfo(
            name='ShuffleSeed', direction=OptionDirection.OTHER, values=['-1'])
        testset = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.INPUT,
            values=['DATASET'])
        return [dataset, size, seed, testset]

    def play(self, bucket: Bucket):
        dataset: tf.data.Dataset = bucket[self.options['SrcDataset']]
        buffersize: int = int(self.options['ShuffleSize'])
        seed: int = int(self.options['ShuffleSeed'])
        if seed == -1:
            dataset = dataset.shuffle(buffersize)
            LogUtil.debug('dataset will be shuffled with buffersize {}'.format(
                buffersize))
        else:
            dataset = dataset.shuffle(buffersize, seed=seed)
            LogUtil.debug(
                'dataset will be shuffled with buffersize {} and seed {}'.
                format(buffersize, seed))
        bucket[self.options['DestDataset']] = dataset


class DatasetAugument(KitPart):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        dataset = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.INPUT,
            values=['DATASET'])
        rotation = OptionInfo(
            name='Rotation',
            direction=OptionDirection.OTHER,
            values=['ON,OFF'])
        return [dataset]


class DatasetBooster(KitPart):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    print(tf.__version__)
    LogUtil.init()
    player = KitPlayer()
    player.set_playlist(json_file='./test_play.json')
    player.play()
'''
* GUIで設定する。
* BlockPlay
* (Electron<->)Flask (),
* 
* 学習中はTensorBoadを表示する。
* 中間データはcsv.
* 使用できるモジュールリストがほしい。
* モジュールのオプションがほしい。

* トレーニング・ブロックを作る
    * 過去トレーニングログからコピーする。
    * 最終的な設定値を送り、実行する。
    * 実行中のステータスを確認したい。
    * 学習を中断したい。(プロセスの実行）
    * 学習履歴を列挙したい。

* 評価ブロック
    * 評価フロー
    * 評価結果
        ROC/フィルタ

* バケットチェック
    バケットにないものをアクセスする


* オプションのネスト、
    Hoge.xxxx記法であること、
    ON/OFFの下に記載のこと。

'''
