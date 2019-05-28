import tensorflow as tf
import os
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket


class AutoEncoderModelCompile(ErmineUnit):
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
        
        return [src, dest, optimizer, lr]

    def run(self, bucket: Bucket):
        model: tf.keras.Model = bucket[self.options['SrcModel']]
        lr = float(self.options['LearningRate'])
        if self.options['Optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(lr=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        bucket[self.options['DestModel']] = model


class AutoEncoderModelTrain(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        model = OptionInfo(
            name='Model',
            direction=OptionDirection.INPUT,
            values=['MODEL'])
        
        train = OptionInfo(
            name='TrainDataset',
            direction=OptionDirection.INPUT,
            values=['TRAIN_DATASET'])

        train_size  = OptionInfo(
            name='TrainDatasetSize',
            direction=OptionDirection.INPUT,
            values=['TRAIN_DATASET_SIZE'])

        validation = OptionInfo(
            name='ValidaitonDataset',
            direction=OptionDirection.INPUT,
            values=['VALIDATION_DATASET'])

        validation_size  = OptionInfo(
            name='ValidationDatasetSize',
            direction=OptionDirection.INPUT,
            values=['VALIDATION_DATASET_SIZE'])
        
        tensorboard = OptionInfo(
            name='TensorBoard',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])

        trial_name = OptionInfo(
             name='TraialName',
             direction=OptionDirection.PARAMETER,
             values=['traial1'])

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

        check_point = OptionInfo(
            name='ModelCheckpoint',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])

        check_point_save_best = OptionInfo(
            name='ModelCheckpoint.SaveBestOnly',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])

        check_point_save_weigths_only = OptionInfo(
            name='ModelCheckpoint.SaveWeightsOnly',
            direction=OptionDirection.PARAMETER,
            values=['True', 'False'])
        
        check_point_file_path = OptionInfo(
            name='ModelCheckpoint.FilePath',
            direction=OptionDirection.PARAMETER,
            values=['weights.hdf5'])
            # values=['weights-{epoch:02d}-{val_loss:.2f}.hdf5'])
        
        max_epoch = OptionInfo(
            name='MaxEpoch',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        
        batch = OptionInfo(
            name = 'BatchSize',
            direction=OptionDirection.PARAMETER,
            values=['100']
        )

        return [model, train, train_size, validation, validation_size, tensorboard, trial_name, early_stopping, early_stopping_patience, early_stopping_monitor, check_point, check_point_save_best, check_point_save_weigths_only, check_point_file_path, max_epoch, batch]

    def run(self, bucket: Bucket):

        # sess = tf.Session()
        # sess.as_default()

        model: tf.keras.Model = bucket[self.options['Model']]
        model.summary()
        train_dataset: tf.data.Dataset = bucket[self.options['TrainDataset']]
        train_size: int = int(bucket[self.options['TrainDatasetSize']])
        validation_dataset: tf.data.Dataset = bucket[self.options['ValidaitonDataset']]
        validation_size: int = int(bucket[self.options['ValidationDatasetSize']])

        batch_size: int = int(self.options['BatchSize'])
        max_epoch: int = int(self.options['MaxEpoch'])

        steps_per_epoch = train_size//batch_size
        validation_steps = validation_size//batch_size

        def map_fn(x, y):
            return (x,x)

        train_dataset = train_dataset.map(map_func=map_fn).repeat().batch(batch_size)
        validation_dataset = validation_dataset.map(map_func=map_fn).repeat().batch(batch_size)

        print(train_dataset)
        print(validation_dataset)

        callbacks = []
        if bool(self.options['TensorBoard']):
            # self.config['globals']['TASKDIR'] = seq_dir
            log_dir = self.globals['TASKDIR'] + os.path.sep + self.options['TraialName'] +  os.path.sep + 'log'
            if os.path.exists(log_dir) is not True:
                os.makedirs(log_dir)
                print('log dir is ' + log_dir)
            tf_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, update_freq='epoch')
            callbacks.append(tf_cb)
            print("Append Tensorboard")

        if bool(self.options['EarlyStopping']):
            patience = int(self.options['EarlyStopping.Patience'])
            monitor = self.options['EarlyStopping.Monitor']
            es_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode='auto')
            callbacks.append(es_cb)
        
        if bool(self.options['ModelCheckpoint']):
            best = bool(self.options['ModelCheckpoint.SaveBestOnly'])
            weigths = bool(self.options['ModelCheckpoint.SaveWeightsOnly'])
            filepath = self.globals['TASKDIR'] + os.path.sep + self.options['TraialName'] +  os.path.sep + self.options['ModelCheckpoint.FilePath']

            sv_cb = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=best, save_weights_only=weigths, verbose=1)
            callbacks.append(sv_cb)

        model.fit(train_dataset, validation_data=validation_dataset, callbacks=callbacks, epochs=max_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

