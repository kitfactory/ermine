from typing import List
from .. base import Bucket, OptionInfo, OptionDirection, ErmineUnit


class FooUnit(ErmineUnit):
    def __init__(self):
        super().__init__()

    @staticmethod
    def prepare_option_infos() -> List[OptionInfo]:
        o = OptionInfo(
            name='message_key',
            direction=OptionDirection.OUTPUT,
            values=['Message'])
        g = OptionInfo(
            name = 'task',
            direction=OptionDirection.INPUT,
            values=['$GLOBAL.Task$'])
        return [o,g]

    def run(self, bucket: Bucket):
        print('foo unit options ', self.options)
        bucket[self.options['message_key']] = 'HelloWorld'
        print( 'option[task]' , self.options['task'])
        bucket['task'] = self.options['task']
        print("FooUnit.run()")
        print('FooUnit.task',self.options['task'])

