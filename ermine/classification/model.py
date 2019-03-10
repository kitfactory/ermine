import tensorflow as tf
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket


class SimpleCNN(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        model = OptionInfo(
            name='Model',
            direction=OptionDirection.OUTPUT,
            values=['MODEL'])
        size = OptionInfo(
            name='Size',
            direction=OptionDirection.PARAMETER,
            values=['28'])
        channel = OptionInfo(
            name='Channel',
            direction=OptionDirection.PARAMETER,
            values=['1'])
        cls = OptionInfo(
            name='Class',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        return [model, size, channel, cls]

    def run(self, bucket: Bucket):
        size = int(self.options['Size'])
        channel = int(self.options['Channel'])
        cls = int(self.options['Class'])

        kernel_size = (3, 3)
        max_pool_size = (2, 2)
        input_layer = tf.keras.Input([size, size, channel])

        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(input_layer)
        cnn = tf.keras.layers.Dropout(0.1)(cnn)
        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
#        cnn = tf.keras.layers.Dropout(0.1)(cnn)
#        cnn = tf.keras.layers.MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

        fc = tf.keras.layers.Flatten()(cnn)
        fc = tf.keras.layers.Dense(1024, activation='relu')(fc)
        softmax = tf.keras.layers.Dense(cls, activation='softmax')(fc)
        model = tf.keras.Model(inputs=input_layer, outputs=softmax)
        bucket[self.options['Model']] = model


class ModelCompile(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='SrcModel',
            direction=OptionDirection.INPUT,
            values=['MODEL'])
        dest = OptionInfo(
            name='DestModel',
            direction=OptionDirection.OUTPUT,
            values=['MODEL'])
        optimizer = OptionInfo(
            name='Optimizer',
            direction=OptionDirection.OUTPUT,
            values=['SGD,Adam'])
        lr = OptionInfo(
            name='LearningRate',
            direction=OptionDirection.OUTPUT,
            values=['0.01'])

        return [src, dest, optimizer, lr]

    def run(self, bucket: Bucket):
        model: tf.keras.Model = bucket[self.options['SrcModel']]
        lr = float(self.options['LearningRate'])
        if self.options['Optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(lr=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        bucket[self.options['DestModel']] = model


class ModelTrain(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        model = OptionInfo(
            name='Model',
            direction=OptionDirection.INPUT,
            values=['MODEL'])
        
        dataset = OptionInfo(
            name='Dataset',
            direction=OptionDirection.INPUT,
            values=['DATASET'])
        
        tensorboard = OptionInfo(
            name='TensorBoard',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])

        tensorboard_dir = OptionInfo(
            name='TensorBoard.LogDir',
            direction=OptionDirection.PARAMETER,
            values=['./log'])

        early_stopping = OptionInfo(
            name='EarlyStopping',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])
        
        early_stopping_patience = OptionInfo(
            name='EarlyStopping.Patience',
            direction=OptionDirection.PARAMETER,
            values=['5'])

        early_stopping_monitor = OptionInfo(
            name='EarlyStopping.Monitor',
            direction=OptionDirection.PARAMETER,
            values=['loss', 'val_loss'])

        cls = OptionInfo(
            name='Class',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        
        max_epoch = OptionInfo(
            name='MaxEpoch',
            direction=OptionDirection.PARAMETER,
            values=['10'])

        return [model, dataset, tensorboard, tensorboard_dir, early_stopping, early_stopping_patience, early_stopping_monitor, cls, max_epoch]

    def __input_fn(self, dataset) -> tf.data.Dataset:
        return dataset.make_one_shot_iterator().get_next()
 
    def run(self, bucket: Bucket):
        model: tf.keras.Model = bucket[self.options['Model']]
        dataset: tf.data.Dataset = bucket[self.options['Dataset']]
        self.cls = int(self.options['Class'])

        def map_fn(image, label):
            '''Preprocess raw data to trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), 10)
            # y = tf.reshape(y, (1, 10))
            return ({'input_1': x}, y)

        dataset = dataset.map(map_fn).batch(50).repeat().prefetch(tf.contrib.data.AUTOTUNE)
        # trainset = dataset.take(54000).skip(6000)
        # valset = dataset.skip(54000).take(6000)

        callbacks = []
        if bool(self.options['TensorBoard']):
            log_dir = self.options['TensorBoard.LogDir']
            tf_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
            callbacks.append(tf_cb)
        if bool(self.options['EarlyStopping']):
            patience = int(self.options['EarlyStopping.Patience'])
            monitor = self.options['EarlyStopping.Monitor']
            es_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode='auto')
            callbacks.append(es_cb)


#       dataset = dataset.map(self.__one_hot)
        # model.fit(trainset, validation_data=valset, callbacks=callbacks, epochs=10, steps_per_epoch=1080, validation_steps=120)

        # KerasモデルをEstimatorに変換する
        print(model.input_names)
        print(model.output_names)
        est: tf.estimator.Estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='./estimator')
        est.train(input_fn=lambda: self.__input_fn(dataset), steps=120)


'''
tf.keras.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs
)
'''