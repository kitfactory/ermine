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
            values=['DATASET'])
        
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224'])
        
        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['DATASET'])

        return [src, size, dest]

    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]
        size = int(self.options['Size'])
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
            direction=OptionDirection.INPUT,
            values=['DATASET'])

        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224'])

        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['DATASET'])

        return [src, size, dest]

    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]
        size = int(self.options['Size'])
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
        size = int(self.options['Size'])
        def resize(x, y):
            x = tf.image.resize(x, size=(size, size))
            return x, y
        dataset = dataset.map(resize)
        bucket[self.options['DestDataset']] = dataset






'''
tf.image.random_hue(
    image,
    max_delta,
    seed=None
)


def fn1():
    distorted_image=tf.image.random_contrast(image)
    distorted_image=tf.image.random_brightness(distorted_image)
    return distorted_image
def fn2():
    distorted_image=tf.image.random_brightness(image)
    distorted_image=tf.image.random_contrast(distorted_image)
    return distorted_image

# Uniform variable in [0,1)
p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
pred = tf.less(p_order, 0.5)

distorted_image = tf.cond(pred, fn1, fn2)
'''
