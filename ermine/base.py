from typing import List, Dict
from enum import Enum
from abc import ABCMeta, abstractmethod

from os.path import expanduser
import subprocess
import os
import json
import tensorflow as tf


class OptionDirection(Enum):
    INPUT = 1
    OUTPUT = 2
    PARAMETER = 3


class ErmineException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OptionInfo():
    def __init__(self,
                 json: str = None,
                 name: str = None,
                 direction: OptionDirection = OptionDirection.PARAMETER,
                 values: List[str] = ['']):
        if json is not None:
            dict: Dict = json.load(json)
            self.name = dict['name']
            direction: Dict[int, OptionDirection] = {
                1: OptionDirection.INPUT,
                2: OptionDirection.OUTPUT,
                3: OptionDirection.PARAMETER
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


class Bucket(Dict):
    def __init__(self):
        super().__init__()


class ErmineUnit(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.option_infos: List[OptionInfo] = self.prepare_option_infos()
        self.options: Dict[str, str] = self.__prepare_options_with_default()

    def get_module_name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def prepare_option_infos() -> List[OptionInfo]:
        pass

    @staticmethod
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
    def run(self, bucket: Bucket):
        pass


class ErmineState():
    NOT_STARTED: int = 0
    EXECUTING: int = 1
    DONE: int = 2
    ERROR: int = 3

    def __init__(self, name: str):
        self.state = ErmineState.NOT_STARTED
        self.name = name
        self.description = ''

    def get_name(self):
        return self.name

    def set_state(self, state: int):
        self.state = state

    def get_state(self) -> int:
        return self.state

    def set_description(self, description: str):
        self.description = description

    def get_description(self) -> str:
        return self.description


class ErmineSubprocessRunner():
    def __init__(self,):
        super().__init__()

    def run(self, id: str):
        com: str = 'ermine-runner ' + HomeFile.get_home_dir() + os.path.sep + 'run_{}.json'.format(id)
        self.process = subprocess.Popen(com.split())

    def abort(self):
        self.process.terminate()


class ErmineRunner():
    def __init__(self, web: bool = False):
        super().__init__()
        self.json_str: str
        self.config: Dict[str, str]
        self.units: List[ErmineUnit] = []
        self.status: List[ErmineState] = []
        self.running = False
        self.web = web
  
    def __stanby_execute(self):
        try:
            unit_list = self.config['units']

            for block_config in unit_list:
                name = block_config['name']
                name = name.rsplit('.', 1)
                print(name)
                mod = __import__(name[0], fromlist=["name"])
                class_def = getattr(mod, name[1])
                instance = class_def()
                # block_config['options']['GLOBAL'] = global_option
                instance.set_options(block_config['options'])
                self.units.append(instance)
                self.status.append(ErmineState(name))
        except Exception as exception:
            raise ErmineException(exception)

    def get_config(self) -> Dict[str, str]:
        return self.config

    def set_config(self, json_str: str = None, json_file: str = None, update_str: bool = True):
        try:
            if json_str is None and json_file is None:
                raise ErmineException(
                    'No specified configuration.'
                )

            if json_file is not None:
                with tf.gfile.GFile(json_file, 'r') as f:
                    json_str = f.read()

            if json_str is not None:
                if update_str:
                    self.json_str = json_str
                self.config = json.loads(json_str, encoding='utf-8')

            if 'globals' in self.config and update_str:
                global_options = self.config['globals']
                global_dict: Dict[str, str] = {}
                for k in global_options:
                    global_dict['$GLOBAL.'+k+'$'] = global_options[k]
                self.overwrite_units_block_config(global_dict)

        except Exception as exception:
            raise ErmineException(exception)

    def overwrite_units_block_config(self, overwrite: Dict[str, str]):
        units = self.config['units']
        s = json.dumps(units)
        for k in overwrite:
            s = s.replace(k, overwrite[k])
        new_options = json.loads(s)
        self.config['units'] = new_options
        self.set_config(json_str=json.dumps(self.config), update_str=False)

    def run(self, bucket: Bucket = None) -> Bucket:
        self.__stanby_execute()
        self.running = True
        if bucket is None:
            bucket = Bucket()
        for idx, u in enumerate(self.units):
            self.status[idx].set_state(ErmineState.EXECUTING)
            unit: ErmineUnit = u
            self.__update_status()
            try:
                unit.run(bucket)
                self.status[idx].set_state(ErmineState.DONE)
            except Exception as exception:
                self.running = False
                self.status[idx].set_state(ErmineState.ERROR)
                self.__update_status()
                raise ErmineException(exception)

        self.running = False
        return bucket

    def __update_status(self):
        if self.web is True:
            print('update status')

    def get_status(self) -> List[ErmineState]:
        return self.status


class DatasetLabelStats(ErmineUnit):
    def __init__(self):
        super().__init__()


class DatasetImageStats(ErmineUnit):
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


class HomeFile():
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_home_dir() -> str:
        return expanduser("~") + os.path.sep + '.ermine'

    @staticmethod
    def make_home_dir():
        home = HomeFile.get_home_dir()
        os.mkdir(home)

    @staticmethod
    def load_home_file(file) -> str:
        file = HomeFile.get_home_dir() + os.path.sep + file
        if os.path.exists(file):
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                return f.readlines()
        else:
            return None

    @staticmethod
    def save_home_file(file: str, conetents: str):
        file = HomeFile.get_home_dir() + os.path.sep + file
        with open(file, mode='w', encoding='utf-8',  newline='\n') as f:
            f.write(conetents)
    
    @staticmethod
    def get_seq_dir(seq) -> str:
        seq_dir: str = HomeFile.get_home_dir() + os.path.sep + 'task' + os.path.sep + seq
        return seq_dir

    @staticmethod
    def make_seq_dir(seq):
        os.makedirs(HomeFile.get_seq_dir(seq))


class Sequence():
    instance = None
    SEQUENCE_FILE = 'sequence'

    def __init__(self):
        super().__init__()
        val: str = HomeFile.load_home_file(Sequence.SEQUENCE_FILE)
        self.seqence = 0
        if val is not None:
            self.sequence = int(val)

    @staticmethod
    def get_sequcene() -> int:
        if Sequence.instance is None:
            Sequence.instance = Sequence()
        else:
            Sequence.instance.seqence = Sequence.instance.seqence + 1
            HomeFile.save_home_file(Sequence.SEQUENCE_FILE, str(Sequence.instance.seqence))
        return Sequence.instance.seqence


class JsonInitialParameter(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        o = OptionInfo(
            name='json_path',
            direction=OptionDirection.INPUT,
            values=['param.json'])
        return [o]

    def run(self, bucket: Bucket):
        pass


class ShuffleDataset(ErmineUnit):
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
            direction=OptionDirection.PARAMETER,
            values=['100000'])
        seed = OptionInfo(
            name='ShuffleSeed', direction=OptionDirection.PARAMETER, values=['-1'])
        testset = OptionInfo(
            name='DestDataset',
            direction=OptionDirection.INPUT,
            values=['DATASET'])
        return [dataset, size, seed, testset]

    def run(self, bucket: Bucket):
        pass


class DatasetAugument(ErmineUnit):
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
            direction=OptionDirection.PARAMETER,
            values=['ON,OFF'])
        return [dataset]

    def run(self, bucket: Bucket):
        pass


class DatasetBooster(ErmineUnit):
    def __init__(self):
        super().__init__()


class WooUnit(ErmineUnit):
    def __init__(self):
        super().__init__()

    @staticmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        o = OptionInfo(
            name='message_key',
            direction=OptionDirection.OUTPUT,
            values=['Message'])
        return [o]

    def run(self, bucket: Bucket):
        bucket[self.config['message_key']] = 'HelloWorld'
        print("FooUnit.run()")



if __name__ == '__main__':
    print(tf.__version__)
    LogUtil.init()
    player = ErmineRunner()
    player.set_playlist(json_file='./test_play.json')
    player.run()
'''
* GUIで設定する。
* BlockPlay
* (Electron<->)Flask (),


* 学習中はTensorBoadを表示する。



* 中間データはcsv.
* 使用できるモジュールリストがほしい。
* モジュールのオプションがほしい。

* トレーニング・ブロックを作る.

* 過去トレーニングログからコピーする。
* 最終的な設定値を送り、実行する。
* 実行中のステータスを確認したい。
* 学習を中断したい。(プロセスの実行）
* 学習履歴を列挙したい。

* ModelのEstimator化
* Tensorboardのログの位置
    

* 評価ブロック
* 評価フロー : 
* 評価結果   : 
ROC/フィルタ :

* バケットチェック
  バケットにないものをアクセスする


* オプションのネスト、
Hoge.xxxx記法であること、
ON/OFFの下に記載のこと。


* 共通の変数を使用する。
* global->$Global.**$



* 生成した値で動かす。
* templateファイル

* task data
    - タスク結果が見られる。
    - 結果を見るには？
        - 検証フロー




Estimator API



Structure of a pre-made Estimators program
 Write one or more dataset importing functions
 feature columnを渡す

Custom Estimators

kerasモデルをEstimator化するには、
compile後にmodel_to_estimator()する。


keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)


tf.keras.estimator.model_to_estimator(
    keras_model=None, # 必要
    keras_model_path=None,
    custom_objects=None,
    model_dir=None,   # 必要
    config=None # 必要
)


mage_col = tf.feature_column.numeric_column('pixels', shape=[image_width * image_height])


'''
