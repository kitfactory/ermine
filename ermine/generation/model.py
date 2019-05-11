import tensorflow as tf
import os
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket

class SimpleAutoEncoder(ErmineUnit):

    def __init__(self):
        super().__init__()

    @classmethod
    def prepared_parameters(cls):
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
        return [model, size, channel]

    def run(self, bucket:Bucket):
        size = int(self.options['Size'])
        channel = int(self.options['Channel'])
        model_dest = self.options['Model']

        inputs = tf.keras.layers.Input(shape=(size,size,channel))
        mid1 = tf.keras.layers.Dense(size*size*channel/4, activation='relu')(inputs)
        mid2 = tf.keras.layers.Dense(size*size*channel/16, activation='relu')(mid1)
        mid3 = tf.keras.layers.Dense(size*size*channel/64, activation='relu')(mid2)
        mid4 = tf.keras.layers.Dense(size*size*channel/16, activation='relu')(mid3)
        mid5 = tf.keras.layers.Dense(size*size*channel/4, activation='relu')(mid4)
        last = tf.keras.layers.Dense(size*size*channel)(mid5)
        
        reshape = tf.keras.layers.Resaphe((size, size, channel))(last)
        model = tf.keras.Model(inputs=inputs, outputs=reshape)
        bucket[model_dest] = model



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
            direction=OptionDirection.PARAMETER,
            values=['SGD,Adam'])
        lr = OptionInfo(
            name='LearningRate',
            direction=OptionDirection.PARAMETER,
            values=['0.01'])
        
        return [src, dest, optimizer, lr, loss]

    def run(self, bucket: Bucket):
        model: tf.keras.Model = bucket[self.options['SrcModel']]
        lr = float(self.options['LearningRate'])
        if self.options['Optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(lr=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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


        logdir: str = "." + os.path.sep+ "log"

        tensorboard_dir = OptionInfo(
            name='TensorBoard.LogDir',
            direction=OptionDirection.PARAMETER,
            values=[logdir])

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
            values=['5'])

        return [model, dataset, tensorboard, tensorboard_dir, early_stopping, early_stopping_patience, early_stopping_monitor, cls, max_epoch]

    def __input_fn(self, dataset) -> tf.data.Dataset:
        return dataset.make_one_shot_iterator().get_next()
 
    def run(self, bucket: Bucket):

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.compat.v1.Session(config=config)

        sess = tf.Session()
        sess.as_default()

        model: tf.keras.Model = bucket[self.options['Model']]
        dataset: tf.data.Dataset = bucket[self.options['Dataset']]
        
        print(dataset)

        def short_map(image, label):
            x = tf.reshape(image, (28,28,1))
            y = tf.one_hot(tf.cast(label, tf.uint8), 10)
            return (x, y)


        # dataset = dataset.map(short_map).batch(100).repeat()
        # print(dataset)
        # model.fit(dataset, epochs=10, steps_per_epoch=600)

        # print("End!!")
        # exit()


        self.cls = int(self.options['Class'])

        def map_fn(image, label):
            '''Preprocess raw data to trainable input. '''
            #  x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), self.cls)
            # y = tf.reshape(y, (1, self.cls))
            # return ({'input_1': x}, y)
            return (image, y)

        dataset = dataset.map(map_fn) #.batch(100).repeat() #.prefetch(1000)
        trainset = dataset.take(54000).skip(6000).batch(50).repeat()
        valset = dataset.skip(54000).take(6000).batch(50).repeat()
        print("trainset", trainset)
        print("valset", valset)

        callbacks = []
        if bool(self.options['TensorBoard']):
            log_dir = self.options['TensorBoard.LogDir']
            tf_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, update_freq='epoch')
            callbacks.append(tf_cb)
            print("Append Tensorboard")

        if bool(self.options['EarlyStopping']):
            patience = int(self.options['EarlyStopping.Patience'])
            monitor = self.options['EarlyStopping.Monitor']
            es_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode='auto')
            callbacks.append(es_cb)
        
        if bool(self.options['ModelSaveCheckpoint']):
            sv_cb = tf.keras.callbacks.ModelSaveCheckpoint()
            callbacks.append(sv_cb)

#       dataset = dataset.map(self.__one_hot)
        model.fit(trainset, validation_data=valset, callbacks=callbacks, epochs=10, steps_per_epoch=1080, validation_steps=120)
