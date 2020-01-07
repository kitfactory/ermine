import re
from abc import ABC, abstractmethod
from typing import Dict
from typing import List
from jsonschema import validate, ValidationError
from jinja2 import Template,Environment,DictLoader

import optuna
import copy
import json
from .log import ErmineLogger


class WorkingBucket(Dict):
    def __init__(self):
        pass

class ErmineModule2(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_module_description(self,lang:str="en_US")->str:
        """ get module description
        
        Returns:
            str: module description
        """
        pass

    @abstractmethod
    def get_setting_schema(self,lang:str="en_US")->Dict:
        """ get setting parameter infos as json schema
        
        Args:
            lang (str): get setting info as json schema in each language
        """
        raise ErmineException("Not Implemented")

    @abstractmethod
    def set_config(self,config:Dict):
        """set configuration to the module
        
        Args:
            config (Dict): 
        
        Raises:
            ErmineException: when config has a wrong value.
        """
        raise ErmineException("Not Implemented")

    @abstractmethod
    def execute(self, bucket:WorkingBucket)->None:
        """execute module functions
        
        Args:
            bucket (WorkingBucket): working object that contains 
        
        Raises:
            ErmineException: [description]
        """
        raise ErmineException("Not Implemented")


class TestModule2(ErmineModule2):

    def get_setting_schema(self,lang:str):
        pass

    def get_module_name(self)->str:
        return "testmodule2"

    def set_config(self,config:Dict):
        print("set_config")
    
    def execute(self, bucket:WorkingBucket)->None:
        print("execute")


class ModuleExecutor():
    def __init__(self):
        self.modules = []

    def prepare_modules(self):
        pass

    def execute(self,config:Dict):
        process = config["Process"]
        for p in process:
            pass

class OptunaExecutor():
    """config executor with optuna
    
    Returns:
        [type]: [description]
    """

    REPLACE_KEYWORD = "$"

    def __init__(self):
        self.logger:ErmineLogger = ErmineLogger.get_instance()

    def __get_config(self):
        pass

    def __config_to_module_instance(self,process_config:Dict)->List[ErmineModule2]:
        pass

    def __is_optuna_value(self, value:str)->bool:
        # discrete_uniform(x,x,x)
        # int(x,x)
        # loguniform(x,x)
        # category([x,x,x,x,])



        # suggest_discrete_uniform(low,high,step) 間の空いた値
        # suggest_int(low,high)　int
        # suggest_loguniform loguniform(a,b)
        # suggest_uniform  uniform(a,b)
        # suggest_categorical(['a','b']) categ
        pass

    def generate_params(self, config:Dict)->Dict:
        pass

    def search_nest(self,subdict:Dict)->Dict:
        pass

    def generate_optuna_template(self,config:Dict)->(Dict[str, str],Dict[str,str]):
        """find optuna parameters from config values and generate template dictionary and value dictionary.

        Args:
            self ([type]): [description]
            List ([type]): [description]
        
        Returns:
            Dict[type]: [description]
            Dict[str,str] : key - module.variable.optuna_suggest_config, optuna_suggest_config
        """

        template_dict = copy.deepcopy(config)
        
        process = template_dict["Process"]
        templated_process = []
        optuna_val = []
        optuna_val_dict = {}

        for p in process:
            module = p["Module"]
            setting = p["Setting"]
            self.logger.debug(module)
            for k in setting.keys():
                self.logger.debug("key : " + k)
                if isinstance(setting[k], str):
                    self.logger.debug("challenge match:" + setting[k])
                    pattern = "uniform\(.*\)|discrete_uniform\(.*\)|loguniform\(.*\)|int\(.*\)|categorical\(.*\)" # "loguniform(.*)|uniform(.*)|discrete_uniform(.*)|"
                    match = re.match(pattern,setting[k])
                    if match is not None:
                        self.logger.debug("match:" + match.group(0))
                        variable_str = module+ "_"+k + "_" + setting[k]
                        variable_str = variable_str.replace("(","_")
                        variable_str = variable_str.replace(")","")
                        variable_str = variable_str.replace("[","")
                        variable_str = variable_str.replace("]","")
                        variable_str = variable_str.replace(",","_")
                        variable_str = variable_str.replace("\"","")
                        self.logger.debug("variable_str:" +variable_str)
                        setting[k] = OptunaExecutor.REPLACE_KEYWORD + variable_str + OptunaExecutor.REPLACE_KEYWORD
                        optuna_val_dict[variable_str] = match.group(0)

            p["Setting"] = setting
            templated_process.append(p)
        
        template_dict["Process"] = templated_process
        self.logger.debug(template_dict)
        self.logger.debug(optuna_val_dict)
        return template_dict,optuna_val_dict


    def generate_trial_config(self, template:Dict, optuna_vals:Dict)->Dict:
        """generate traial config with optuna suggested values.
        
        Args:
            config (Dict): config of the study
            optuna_vals (Dict): values generated by optuna
        """
        json_str = json.dumps(template,indent=4)
        self.logger.debug(json_str)
        loader:DictLoader = DictLoader({"template":json_str})
        environment:Environment=Environment(loader=loader,trim_blocks=True,block_start_string='<<',block_end_string='>>',variable_start_string='$', variable_end_string='$')
        self.logger.debug(optuna_vals)
        template:Template = environment.get_template("template")
        self.logger.debug("render!!")
        rendered_str = template.render(optuna_vals)
        rendered_json = json.loads(rendered_str)
        return rendered_json
        # environment = jinja2.Environment(loader=loader, trim_blocks=True,block_start_string='@@',block_end_string='@@',variable_start_string='@=', variable_end_string='=@')


    def objective(self, trial:optuna.Trial):
        """otpuna objective function
        
        Args:
            trial (optuna.Trial): traial object of optuna
        
        Returns:
            [type]: [description]
        """

        logger = ErmineLogger.get_instance()
        logger.debug("objective")
        optuna_dict = {}
        template = self.template
        optuna_params = self.optuna_params
        logger.debug(optuna_params)

        for p_key in optuna_params:
            p = optuna_params[p_key]
            logger.debug("p in optuna key "+ p_key + " , " + "val "+ p)
            if( p.startswith("uniform")):
                uni_pattern = "uniform\((.*),(.*)\)"
                # logger.debug("check unipattern " + p)
                matchobj = re.match(uni_pattern,p)
                low = float(matchobj.group(1))
                high = float(matchobj.group(2))
                # print(matchobj.group(0) + "," + matchobj.group(1))
                v = trial.suggest_uniform(p,low,high)
                optuna_dict[p_key] = str(v)
            elif p.startswith("loguniform"):
                loguni_pattern = "loguniform\((.*),(.*)\)"
                # logger.debug("check log unipattern " + p)
                matchobj = re.match(loguni_pattern,p)
                low = float(matchobj.group(1))
                high = float(matchobj.group(2))
                # print(matchobj.group(0) + "," + matchobj.group(1))
                v = trial.suggest_loguniform(p,low,high)
                optuna_dict[p_key] = str(v)
            elif p.startswith("categorical"):
                category_pattern = "categorical\((\[.*\])\)"
                matchobj = re.match(category_pattern,p)
                str_array = matchobj.group(1)
                # print(str_array)
                json_array = json.loads(str_array)
                # print(json_array)
                v = trial.suggest_categorical(p,json_array)
                optuna_dict[p_key]=v
            elif p.startswith("int"):
                int_pattern = "int\((.*),(.*)\)"
                # logger.debug("check int unipattern " + p)
                matchobj = re.match(int_pattern,p)
                low = float(matchobj.group(1))
                high = float(matchobj.group(2))
                # print(matchobj.group(0) + "," + matchobj.group(1))
                v = trial.suggest_loguniform(p,low,high)
                optuna_dict[p_key] = str(v)
            elif p.startswith("discrete_uniform"):
                disc_uni_pattern = "discrete_uniform\((.*),(.*),(.*)\)"
                # logger.debug("check discrete unipattern " + p)
                matchobj = re.match(disc_uni_pattern,p)
                logger.debug(matchobj)
                low = float(matchobj.group(1))
                high = float(matchobj.group(2))
                q = float(matchobj.group(3))
                # print(matchobj.group(0) + "," + matchobj.group(1)+"," + matchobj.group(2))
                v = trial.suggest_discrete_uniform(p,low,high,q)
                optuna_dict[p_key] = str(v)

        self.logger.debug("optuna trial values")
        self.logger.debug(optuna_dict)

        trial_config = self.generate_trial_config(template,optuna_dict)

        self.logger.debug(trial_config)
        
        # print(optuna_dict)
        bucket:WorkingBucket = self.execute(trial_config)
        if "Result" in bucket:
            return bucket["Result"]
        else:
            return 0

    def optuna_execution(self, study_name:str, trials:int, config:Dict):
        self.logger.info("start optuna {} trials.".format(trials))
        study:optuna.Study = optuna.create_study(study_name=study_name)
        template, optuna_params = self.generate_optuna_template(config)
        self.template = template
        self.optuna_params = optuna_params
        study.optimize(self.objective, n_trials=trials)
        self.logger.info("all optuna trials have been done.")

    def execute(self, config:Dict)->WorkingBucket:

        process = config["Process"]
        process_module = []

        for m in process:
            fullname = m['Module']
            setting = m['Setting']

            tmp = fullname.rsplit('.', 1)
            if len(tmp) == 2:
                self.logger.debug("pkg :"+pkg + ",name:"+name)
                pkg = tmp[0]
                name = tmp[1]
                mod = __import__(pkg, fromlist=[name])
                class_def = getattr(mod, name)
                instance:ErmineModule2 = class_def()
            else:
                self.logger.debug("fullname:"+ fullname)
                mod = __import__("__main__", fromlist=[fullname])
                class_def = getattr(mod, fullname)
                instance:ErmineModule2 = class_def()

            instance.set_config(setting)
            process_module.append(instance)
        
        bucket = WorkingBucket()
        for m in process_module:
            mod:ErmineModule2 = m
            mod.execute(bucket)

        return bucket
