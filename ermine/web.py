from typing import List
from os.path import expanduser, exists

from .base import ErmineUnit
from .base import OptionInfo


class Web():
    def __init__(self):
        self.home = expanduser("~")
        self.units = ['HelloUnit','Hello2Unit']
        if exists(self.home + '.ermine'):
            # load file
            pass
        else:
            pass

    def units(self) -> List[str]:
        return self.units

    def __to_json(self, options: List[OptionInfo])->str:
        ret = []
        for opt in options:
            tmp = {}
            tmp['name'] = opt.get_name()
            tmp['values'] = opt.get_values()
            tmp['direction'] = opt.get_direction()
            tmp['description'] = opt.get_description()
            ret.append(tmp)
        return ret

    def info(self, unit: str) -> str:
        block_class: ErmineUnit = globals()['unit']
        options: List[OptionInfo] = block_class.get_option_infos()
        return self.__to_json(options)

    def execute(self, config: str, sess: str) -> str:
        pass

    def check_status(self) -> str:
        pass
    
    def update_status(self) -> str:
        pass
    
