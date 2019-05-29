from typing import List
import os
import tensorflow as tf
import numpy as np
from .. base import ErmineUnit, OptionInfo, OptionDirection, Bucket
from abc import abstractmethod

class ImageSizeUnit(ErmineUnit):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        train = OptionInfo(
            name='TrainDataset',
            direction=OptionDirection.OUTPUT,
            values=['TRAIN_DATASET'])
        
        validation = OptionInfo(
            name='ValidationDataset',
            direction=OptionDirection.OUTPUT,
            values=['VALIDATION_DATASET'])

        test = OptionInfo(
            name='TestDataset',
            direction=OptionDirection.OUTPUT,
            values=['TEST_DATASET'])

        return [train, validation, test]

    @abstractmethod
    def change_image_size(self, train: tf.data.Dataset, validation: tf.data.Dataset, test: tf.data.Dataset)->(tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        pass

    def run(self, bucket: Bucket):
        train = bucket[self.options['TrainDataset']]
        validation = bucket[self.options['ValidationDataset']]
        test = bucket[self.options['TestDataset']]
        train, validation, test = self.change_image_size(train, validation, test)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['ValidationDataset']] = validation
        bucket[self.options['TestDataset']] = test


class ImageResizeWithCropOrPad(ImageSizeUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        infos = ImageSizeUnit.get_option_infos()
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224'])
        return infos + [size]

    def change_image_size(self, train: tf.data.Dataset, validation: tf.data.Dataset, test: tf.data.Dataset)->(tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        size = int(self.options['Size'])
        def resize(x, y):
            x = tf.image.resize_image_with_crop_or_pad(x, size, size)
            return x, y
        train = train.map(map_func=resize)
        validation = validation.map(map_func=resize)
        test = test.map(map_func=resize)
        return (train,validation,test)

class ImageResize(ImageSizeUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        infos = ImageSizeUnit.get_option_infos()
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224'])
        return infos + [size]

    def change_image_size(self, train: tf.data.Dataset, validation: tf.data.Dataset, test: tf.data.Dataset)->(tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        size = int(self.options['Size'])
        def resize(x, y):
            x = tf.image.resize(x, size, size)
            return x, y
        train = train.map(map_func=resize)
        validation = validation.map(map_func=resize)
        test = test.map(map_func=resize)
        return (train,validation,test)


class ImageCenterCrop(ImageSizeUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        infos = ImageSizeUnit.get_option_infos()
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['224'])
        return infos + [size]

    def change_image_size(self, train: tf.data.Dataset, validation: tf.data.Dataset, test: tf.data.Dataset)->(tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        size = int(self.options['Size'])
        def resize(x, y):
            x = tf.image.central_crop(x, size, size)
            return x, y
        train = train.map(map_func=resize)
        validation = validation.map(map_func=resize)
        test = test.map(map_func=resize)
        return (train,validation,test)






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
