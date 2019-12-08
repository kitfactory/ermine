from abc import ABC

class ErmineModule(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    @classmethod
    def register_module_info(cls, scheme):
        pass

    @abstractmethod
    def execute(self):
        pass
    
class ErmineModuleRunner():
    def __init__(self):
        pass

    def set_config(self, config):
        pass


class ImageClassificationSampleGen(ErmineModule):

    def __init__(self):
        self.train_data_num = 0
        self.validation_data_num = 0
        self.evaluation_data_num = 0

    def count_sample_number(self,config):
        pass

    @abstractmethod
    def __type_of_sample(self):
        return false

    def __split_data(self):
        pass

    def __get_train_dataset(self):
        pass

    @abstractmethod
    def execute(self):
        pass


class ImageClassificationTransformer(ErmineModule):




    
