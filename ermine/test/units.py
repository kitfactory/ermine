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
        return [o]

    def run(self, bucket: Bucket):
        bucket[self.options['message_key']] = 'HelloWorld'
        print("FooUnit.run()")
