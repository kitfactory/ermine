import tensorflow as tf
from .. base import ErmineUnit

class ImageAugument(ErmineUnit):

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        train = OptionInfo(
            name='TrainDataset',
            direction=OptionDirection.OUTPUT,
            values=['TRAIN_DATASET'])

        train_size = OptionInfo(
            name='TrainDatasetSize',
            direction=OptionDirection.INPUT,
            values=['TRAIN_DATASET_SIZE'])
        
        validation = OptionInfo(
            name='ValidationDataset',
            direction=OptionDirection.OUTPUT,
            values=['VALIDATION_DATASET'])

        validation_size = OptionInfo(
            name='ValidationDatasetSize',
            direction=OptionDirection.INPUT,
            values=['VALIDATION_DATASET_SIZE'])


        return [train, train_size, validation, validation_size, test, test_size]

    @abstractmethod
    def augument_images(self, train: tf.data.Dataset, validation: tf.data.Dataset)->(tf.data.Dataset, tf.data.Dataset):
        pass

    def run(self, bucket: Bucket):
        train = bucket[self.options['TrainDataset']]
        validation = bucket[self.options['ValidationDataset']]
        test = bucket[self.options['TestDataset']]
        train, validation, test = self.change_image_size(train, validation, test)
        bucket[self.options['TrainDataset']] = train
        bucket[self.options['ValidationDataset']] = validation
        bucket[self.options['TestDataset']] = test




class GeneralImageAugument(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='SrcDataset',
            direction=OptionDirection.PARAMETER,
            values=['DATASET'])

        brightness = OptionInfo(
            name='RandomBrightness',
            direction=OptionDirection.PARAMETER,
            values=['True','False'])

        brightness_max = OptionInfo(
            name='RandomBrightness.MaxDelta',
            direction=OptionDirection.PARAMETER,
            values=['0.1'])

        brightness_ratio = OptionInfo(
            name='RandomBrightness.Ratio',
            direction=OptionDirection.PARAMETER,
            values=['0.2'])

        contrast = OptionInfo(
            name='RandomContrast',
            direction=OptionDirection.PARAMETER,
            values=['True','False'])

        contrast_lower = OptionInfo(
            name='RandomContrast.Lower',
            direction=OptionDirection.PARAMETER,
            values=['0.1'])
        
        contrast_higher = OptionInfo(
            name='RandomContrast.Heigher',
            direction=OptionDirection.PARAMETER,
            values=['0.2'])

        contrast_ratio = OptionInfo(
            name='RandomContrast.Ratio',
            direction=OptionDirection.PARAMETER,
            values=['0.2'])

        left_right = OptionInfo(
            name='RandomFlipLeftRight',
            direction=OptionDirection.PARAMETER,
            values=['True','False'])

        left_right_ratio = OptionInfo(
            name='RandomContrast.Ratio',
            direction=OptionDirection.PARAMETER,
            values=['0.2'])

        up_down = OptionInfo(
            name='RandomFlipUpDown',
            direction=OptionDirection.PARAMETER,
            values=['True','False'])

        up_down_ratio = OptionInfo(
            name='RandomFlipUpDown.Ratio',
            direction=OptionDirection.PARAMETER,
            values=['0.2'])

        hue = OptionInfo(
            name='RandomHue',
            direction=OptionDirection.PARAMETER,
            values=['True','False'])

        hue_max = OptionInfo(
            name='RandomHue.MaxDelta',
            direction=OptionDirection.PARAMETER,
            values=['0.1'])
        
        hue_ratio = OptionInfo(
            name='RandomHue.Ratio',
            direction=OptionDirection.PARAMETER,
            values=['0.1'])

        dest = OptionInfo(
            name = 'DestDataset',
            direction = OptionDirection.OUTPUT,
            values=['DATASET']
        )

        return [src, brightness, brightness_max, brightness_ratio,
        contrast, contrast_lower, contrast_higher, contrast_ratio,left_right,
        left_right_ratio,up_down,up_down_ratio,hue, hue_max, hue_ratio, dest]


    def run(self, bucket: Bucket):
        dataset = bucket[self.options['SrcDataset']]

        if bool(self.options['RandomBrightness']) is True:
            brightness_ratio = float(self.options['RandomBrightness.Ratio'])
            brightness_max = float(self.options['RandomBrightness.MaxDelta'])

            def random_brightness(x,y):
                rnd = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(rnd, brightness_ratio)
                def true_fn(x):
                    return tf.image.random_brightness(x,max_delta=brightness_max)
                def false_fn():
                    return x
                x = tf.cond(pred, true_fn, false_fn)
                return (x, y)

            dataset = dataset.map(random_brightness)
        
        if bool(self.options['RandomContrast']) is True:
            contrast_ratio = float(self.options['RandomContrast.Ratio'])
            contrast_lower = float(self.options['RandomContrast.Lower'])
            contrast_heigher = float(self.options['RandomContrast.Heigher'])

            def random_contrast(x,y):
                rnd = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(rnd, contrast_ratio)
                def true_fn(x):
                    return tf.image.random_contrast(x, contrast_lower, contrast_heigher)
                def false_fn():
                    return x
                x = tf.cond(pred, true_fn, false_fn)
                return (x, y)

            dataset = dataset.map(random_contrast)


        if bool(self.options['RandomFlipUpDown']) is True:
            up_down_ratio = float(self.options['RandomFlipUpDown.Ratio'])
            def random_up_down(x,y):
                rnd = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(rnd, up_down_ratio * 2)
                def true_fn(x):
                    return tf.image.random_flip_up_down(x)
                def false_fn():
                    return x
                x = tf.cond(pred, true_fn, false_fn)
                return (x, y)

            dataset = dataset.map(random_up_down)

        if bool(self.options['RandomFlipLeftRight']) is True:
            left_right_ratio = float(self.options['RandomFlipLeftRight.Ratio'])
            def random_up_down(x,y):
                rnd = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(rnd, left_right_ratio * 2)
                def true_fn(x):
                    return tf.image.random_flip_left_right(x)
                def false_fn():
                    return x
                x = tf.cond(pred, true_fn, false_fn)
            dataset = dataset.map(random_left_right)


        if bool(self.options['RandomHue']) is True:
            hue_ratio = float(self.options['RandomHue.Ratio'])
            hue_max = float(self.options['RandomHue.MaxDelta'])
            def random_hue(x,y):
                rnd = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(rnd, hue_ratio)
                def true_fn(x):
                    return tf.image.random_hue(x, max_delta=hue_max)
                def false_fn():
                    return x
                x = tf.cond(pred, true_fn, false_fn)
            dataset = dataset.map(random_hue)

        bucket[self.options['DestDataset']] = dataset

