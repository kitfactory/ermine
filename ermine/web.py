from typing import List
import os
from os.path import expanduser, exists
import json

from .base import ErmineUnit
from .base import OptionInfo
from ermine.image.dataset import ImageClassificationDataset

from .process import ProcessUtil


# Flask などの必要なライブラリをインポートする
from flask import Flask, jsonify, redirect, request
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
        cls.process_util = ProcessUtil()

    @classmethod
    def units(cls, data:ErmnDataType, task:ErmineTaskType, unit_type:ErmineUnitType) -> List[str]:
        subclass_list = []
        super_class = None
        return sub
###        if data == ErmnDataType.IMAGE:
###            if task == ErmineTaskType.CLASSIFICATION:
#                if unit_type == ErmineUnitType.DATASET
#                    super_class = ImageClassificationDataset.__class__
#                elif unit_type == ErmineUnitType.TRANSFORM:
#                    super_class = .__class__
#                elif unit_type == ErmineUnitType.AUGUMENT:
#                    super_class = .__class__
#                
#
#
#            for sub in subclass_list:
#                print(sub.__module__,sub.__name__)
#        return cls.unit_list

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
    
    @classmethod
    def execute_tensorboard(cls):
        print('execute tensorboard...')
        cls.process_util.execute_tensorboard()

    @classmethod
    def kill_tensorboard(cls):
        print('killing tensorboard...')
        cls.process_util.kill_tensorboard()
    
    @classmethod
    def is_tensorboard_running(cls):
        return cls.process_util.is_tensorboard_running()

    @classmethod
    def execute_training(cls):
        print('execute training')
        cls.process_util.exec_train()

    @classmethod
    def kill_training(cls):
        print('kill training')
        cls.process_util.kill_training()
    
    @classmethod
    def is_training(cls):
        return cls.process_util.is_training()


    @classmethod
    def execute_evaluation(cls):
        print('execute evaluation')
        cls.process_util.exec_evaluate()

    @classmethod
    def kill_evaluation(cls):
        print('kill evaluation')
        cls.process_util.kill_evaluation()
    
    @classmethod
    def is_evaluating(cls):
        return cls.process_util.is_evaluating()

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

@app.route('/api/exec_training')
def execute_training():
    WebService.execute_training()
    return "Execute Training"

@app.route('/api/is_training')
def is_training():
    return WebService.is_training()

@app.route('/api/kill_training')
def kill_training():
    WebService.kill_training()
    return "Kill Training"

@app.route('/api/exec_evaluation')
def execute_evaluation():
    WebService.execute_evaluation()
    return "Execute Evaluation"

@app.route('/api/kill_evaluation')
def kill_evaluation():
    WebService.kill_evaluation()
    return "Kill Evaluation"

@app.route('/api/is_evaluating')
def is_evaluating():
    return WebService.is_evaluating()

@app.route('/api/tensorboard')
def execute_tensorboard():
    WebService.execute_tensorboard()
    return "Execute Tensorboard"

@app.route('/api/kill_tensorboard')
def kill_tensorboard():
    WebService.kill_tensorboard()
    return "Killed Tensorboard"

@app.route('/api/is_tensorbord_running')
def is_tensorboard_running():
    return WebService.is_tensorboard_running()

@app.route('/api/progress')
def progress():
    return jsonify(WebService.progress())

@app.route('/tensorboard')
def tensorboard():
    return redirect("http://localhost:6006", code=302)

@app.route('/postfile',methods=['GET', 'POST'])
def postfile():
    if request.method == 'POST':
        data = request.data.decode('utf-8')
        data = json.loads(data)
        print(data)
        return 'OK'
    else:
        filepath = request.form['filepath']
        print(filepath)
        return 'OK'

def main():
    print('running web server ....')
    app.debug = True
    WebService.setup()
    app.run(port=7007)

if __name__ == '__main__':
    main()    