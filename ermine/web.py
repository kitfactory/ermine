from typing import List
import os
from os.path import expanduser, exists

from .base import ErmineUnit
from .base import OptionInfo
from ermine.image.dataset import ImageClassificationDataset

# Flask などの必要なライブラリをインポートする
from flask import Flask, jsonify
# , render_template, request, redirect, url_for

from enum import Enum

class ErmnDataType(Enum):
    IMAGE = 1

class ErmineTaskType(Enum):
    CLASSIFICATION = 1
    AUTOENCODER = 2

class ErmineUnitType(Enum):
    DATASET = 1
    TRANSFORM = 2
    AUGUMENT = 3
    TRAIN = 4


class WebService():
    def __init__(self):
        super().__init__()

    @classmethod
    def setup(cls):
        cls.home = expanduser("~")
        cls.unit_list = ['HelloUnit','Hello2Unit']


    @classmethod
    def units(cls, data:ErmnDataType, task:ErmineTaskType, unit_type:ErmineUnitType) -> List[str]:
        subclass_list = []
        super_class = None

        if data == ErmnDataType.IMAGE:
            if task == ErmineTaskType.CLASSIFICATION:
                if unit_type == ErmineUnitType.DATASET
                    super_class = ImageClassificationDataset.__class__
                elif unit_type == ErmineUnitType.TRANSFORM:
                    super_class = .__class__
                elif unit_type == ErmineUnitType.AUGUMENT:
                    super_class = .__class__
                


            for sub in subclass_list:
                print(sub.__module__,sub.__name__)
        return cls.unit_list

    @classmethod
    def __to_json(cls, options: List[OptionInfo])->str:
        ret = []
        for opt in options:
            tmp = {}
            tmp['name'] = opt.get_name()
            tmp['values'] = opt.get_values()
            tmp['direction'] = opt.get_direction()
            tmp['description'] = opt.get_description()
            ret.append(tmp)
        return ret

    @classmethod
    def info(cls, unit: str) -> str:
        block_class: ErmineUnit = globals()['unit']
        options: List[OptionInfo] = block_class.get_option_infos()
        return cls.__to_json(options)

    @classmethod
    def execute(cls, config: str, sess: str) -> str:
        pass

    @classmethod
    def check_status(cls) -> str:
        pass

    @classmethod    
    def update_status(cls) -> str:
        pass
    

contents = os.path.dirname(__file__).split(os.path.sep)[0]  + 'bin' + os.path.sep + 'static'
contents_path = os.path.abspath(contents)
print(contents_path)
app = Flask(__name__, static_folder=contents_path, static_url_path='')


@app.route('/api/info/<unit>')
def info(unit):
    return jsonify(WebService.info(unit))


@app.route('/api/units')
def units():
    unit_list = WebService.units()
    ret = {}
    ret['units'] = unit_list
    return jsonify(ret)


@app.route('/api/exec')
def execute():
    pass

@app.route('/api/progress')
def progress():
    return jsonify(WebService.progress())


def main():
    print('running web server ....')
    app.debug = True
    WebService.setup()
    app.run(port=7007)

if __name__ == '__main__':
    main()    