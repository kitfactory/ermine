from typing import List
import os
import tensorflow as tf
import numpy as np
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket

class ImageCropOrPad(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.PARAMETER,
            values=['Src'])
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224']),
        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['Dest']
        )
        return [src, size, dest]

    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]
        size = int(bucket[self.options['Size']])
        def resize(x, y):
            x = tf.image.resize_image_with_crop_or_pad(x, size, size)
            return x, y
        dataset = dataset.map(resize)
        bucket[self.options['DestDataset']] = dataset


class ImageResize(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.PARAMETER,
            values=['Src'])
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224']),
        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['Dest']
        )
        return [src, size, dest]

    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]
        size = int(bucket[self.options['Size']])
        def resize(x, y):
            x = tf.image.resize(x, size=(size, size))
            return x, y
        dataset = dataset.map(resize)
        bucket[self.options['DestDataset']] = dataset


class ImageCrop(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.PARAMETER,
            values=['Src'])
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224']),
        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['Dest']
        )
        return [src, size, dest]

    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]
        size = int(bucket[self.options['Size']])
        def resize(x, y):
            x = tf.image.resize(x, size=(size, size))
            return x, y
        dataset = dataset.map(resize)
        bucket[self.options['DestDataset']] = dataset

