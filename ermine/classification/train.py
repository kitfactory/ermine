import os
import tensorflow as tf
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket


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
            values=['weights-{epoch:02d}-{val_loss:.2f}.hdf5'])

        clsses = OptionInfo(
            name='Class',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        
        max_epoch = OptionInfo(
            name='MaxEpoch',
            direction=OptionDirection.PARAMETER,
            values=['10'])
        
        batch = OptionInfo(
            name = 'BatchSize',
            direction=OptionDirection.PARAMETER,
            values=['100']
        )

        num_data = OptionInfo(
            name = 'NumberOfData',
            direction=OptionDirection.PARAMETER,
            values=['60000']
        )

        spilit = OptionInfo(
            name = 'ValidationSplit',
            direction=OptionDirection.PARAMETER,
            values=['0.1']
        )

        return [model, dataset, tensorboard, trial_name, early_stopping, early_stopping_patience, early_stopping_monitor, check_point, check_point_save_best, check_point_save_weigths_only, check_point_file_path, clsses, max_epoch, batch, num_data, spilit]

    def __input_fn(self, dataset) -> tf.data.Dataset:
        return dataset.make_one_shot_iterator().get_next()
 
    def run(self, bucket: Bucket):

        sess = tf.Session()
        sess.as_default()

        model: tf.keras.Model = bucket[self.options['Model']]
        dataset: tf.data.Dataset = bucket[self.options['Dataset']]
        batch_size: int = int(self.options['BatchSize'])
        number_of_data: int = int(self.options['NumberOfData'])
        spilit: float = float(self.options['ValidationSplit'])
        max_epoch: int = int(self.options['MaxEpoch'])

        self.clsses = int(self.options['Class'])

        def map_fn(image, label):
            '''Preprocess raw data to trainable input. '''
            #  x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), self.clsses)
            return (image, y)

        train_data_num = int(number_of_data * (1-spilit))
        steps_per_epoch = int(train_data_num/batch_size)
        val_data_num = int(number_of_data * spilit)
        validation_steps = int(val_data_num/batch_size)

        dataset = dataset.map(map_fn)
        trainset = dataset.take(train_data_num).skip(val_data_num).batch(batch_size).repeat()
        valset = dataset.skip(train_data_num).take(val_data_num).batch(batch_size).repeat()

        print("trainset", trainset, "data num" , train_data_num)
        print("valset", valset, "val data num", val_data_num)

        callbacks = []
        if bool(self.options['TensorBoard']):
            #             self.config['globals']['TASKDIR'] = seq_dir
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

            sv_cb = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=best, save_weights_only=weigths)
            callbacks.append(sv_cb)

        model.fit(trainset, validation_data=valset, callbacks=callbacks, epochs=max_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

