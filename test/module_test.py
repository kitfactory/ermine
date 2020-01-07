import unittest
from ermine import ErmineLogger
from ermine import ErmineModule2
from ermine import WorkingBucket
from ermine import OptunaExecutor

from typing import Dict
from typing import List

import sys
import re
import json
import locale

from jsonschema import validate, ValidationError


class TestModule(ErmineModule2):

    def __init__(self):
        self.logger = ErmineLogger.get_instance()

    def get_module_description(self,lang:str="en_US")->str:
        return "Module description"

    def get_setting_schema(self,lang:str="en_US")->Dict:
        self.logger.debug("get setting schema")
        ret = {
            "title":"test module's title",
            "properties":{
                "name":{
                    "type":"string",
                    "description":"name's description"
                },
                "id":{
                    "type":"string",
                    "description":"id's description"
                }
            }            
        }

    def set_config(self,config:Dict):
        self.logger.debug("set config !!")
        self.logger.debug(config)

    def execute(self, bucket:WorkingBucket)->None:
        self.logger.debug("execute !!")


if __name__ == '__main__':

    schema = """
        {
            "title": "PyJsonValidate",
            "type": "object",
            "properties": {
                "ID": {
                    "type":"string"
                },
                "Name": {
                    "type":"string",
                    "pattern":"hoge"
                },
                "HyperParam":{
                    "type":"string",
                    "pattern":"uniform([0-9]*,[0-9]*)"
                }
            }
        }
    """
    
    schema = json.loads(schema)

    item = {
        "id":"1",
        "name": "hoge",
    }

    json_val = """
    {
        "id":"1",
        "name":"hoge"
    }
    """
    validate(schema, item)
    parttern = "(uniform\\([0-9]+\\))|[0-9]+"

    # content = "uniform(20)"
    content = "10"
    print(re.match(parttern,content).group(0))

    #                      "optuna_param":"uniform(10,11)"

    setting_val = {
        "Process":[
            {
                "Module":"TestModule",
                "Setting":{
                     "Name":"hoge",
                     "ID":"xxxxxx",
                     "HyperParam":"uniform(10,20)",
                     "IntHyperParam":"int(5,10)",
                     "CategoricalParam":"categorical([\"a\",\"b\",\"c\"])",
                     "LogUniformParam":"loguniform(10,100)",
                    "DiscreteParam":"discrete_uniform(10,30,5)"
                }
            }
        ]
    }

    loc:locale = locale.getlocale()
    lang = "ja_JP"

    print(loc)

    test = TestModule()
    test.set_config(None)
    test.execute(setting_val)

    template = {
        "x" :"$TestModule_HyperParam_uniform_10_20_$",
    } # {'Process': [{'Module': 'TestModule', 'Setting': {'Name': 'hoge', 'ID': 'xxxxxx', 'HyperParam': '$ TestModule.HyperParam.uniform(10,20) $', 'IntHyperParam': '$TestModule.IntHyperParam.int(5,10)$', 'CategoricalParam': '$TestModule.CategoricalParam.categorical(["a","b","c"])$', 'LogUniformParam': '$TestModule.LogUniformParam.loguniform(10,100)$', 'DiscreteParam': '$TestModule.DiscreteParam.discrete_uniform(10,30,5)$'}}]}
    optuna_vals = {
        "TestModule_HyperParam_uniform_10_20_": "hello"
        # "TestModule.HyperParam.uniform(10,20)":"5.00",
    }

    executor = OptunaExecutor()
    # generated_val=executor.generate_trial_config(template,optuna_vals)

    # new_config, optuna_val = executor.find_optuna_param(setting_val)
    generated_val = executor.optuna_execution(study_name="study",trials=5,config=setting_val)



    print(generated_val) 

    for cls in ErmineModule2.__subclasses__():
        print(cls.__name__)

    # Model
    # Trainer
    

    # executor.generate_setting()
    # executor.execute(setting_val)

    print(sys.path)

        # suggest_discrete_uniform(low,high,step) 間の空いた値
        # suggest_int(low,high)　int
        # suggest_loguniform loguniform(a,b)
        # suggest_uniform  uniform(a,b)
        # suggest_categorical(['a','b']) categ

"""
IMAGE_CLASSIFICATION

DATASET:
    TrainDataset:
    NumTrain:
    TestDataset:
    NumTest:
    Size:28,28,1
    Label:n

    Dataset class must example must contains
    - Image:
    - Label:
    and must put the values described bellow into the bucket.

TFRecordLoad:
    Name:

TFRecordSave:
    Name:
    SaveAs:

Augmentation:
    UnderSample:        
    Crop:
    Resize:
    Shuffle:

Model:
    ResNet50
        PreloadWeight:

Trainer:
    RateOfTrainAndValidation:80,20:
    Loss:Softmax:
    Optimizer:
    LearningRate:
    TensorBoard:
    SaveModel:

Evaluation:
    SaveImage=False
    GradCAM=True
    EvaluationResultName:

{
    "Process":[
        {
            "module":"MnistDataset"
        },
        {

        }

    ]
}


"""