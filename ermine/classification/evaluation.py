import os
import pandas as pd
import tensorflow as tf
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket

class ClassificationPredict(ErmineUnit):
    def __init__(self):
        super().__init__()

    @classmethod
    def prepare_option_infos(self) -> List[OptionInfo]:
        src = OptionInfo(
            name='ModelFile',
            direction=OptionDirection.INPUT,
            values=['MODEL_FILE'])
        data = OptionInfo(
            name='TestDataset',
            direction=OptionDirection.INPUT,
            values=['TEST_DATASET'])
        
        size = OptionInfo(
            name='TestDatasetSize',
            direction=OptionDirection.INPUT,
            values=['TEST_DATASET_SIZE'])
        
        evaluation = OptionInfo(
            name='EvaluationType',
            direction=OptionDirection.PARAMETER,
            values=['OneVursusOne','OneVersusOthers','OneVersusNegativeOthers'])
        
        pickup_class = OptionInfo(
            name='PickupClass',
            direction=OptionDirection.PARAMETER,
            values=['0'])

        pickup_threshold = OptionInfo(
            name='PickupThreshold',
            direction=OptionDirection.PARAMETER,
            values=['0.8'])
    
    
        return [src, data, size, evaluation, pickup_class, pickup_threshold]

    def run(self, bucket: Bucket):
        f = self.options['ModelFile']
        print(f)
        model: tf.keras.Model = tf.keras.models.load_model(f)
        dataset: tf.data.Dataset = bucket[self.options['TestDataset']]
        size: int = bucket[self.options['TestDatasetSize']]
        evaluation = self.options['EvaluationType']
        pickup_class = int(self.options['PickupClass'])
        pickup_threshold = float(self.options['PickupThreshold'])

        dataset = dataset.batch(1)

        ys = []
        yhats = []
        ids = []
        scores = []


        for x, y, i in dataset:
            score = model.predict(x)[0]
            # print(score[0])
            ids.append(i.numpy())
            ys.append(y.numpy().argmax())
            yh = 0
            # print(evaluation)

            if evaluation == 'OneVersusOne':
                yh = score.argmax()
                print(yh)
            elif evaluation == 'OneVersusOthers':
                s = score[pickup_class]
                if s >= pickup_threshold:
                    yh = pickup_class
                else:
                    yh =  -1
            yhats.append(yh)
            scores.append(score)

        frame = pd.DataFrame(
            data={'id':ids, 'y':ys, 'yh':yhats, 'score':scores},
            columns=['id','y', 'yh', 'score']
        )


        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/evaluation.csv')
    


class ClassificationEvaluation(ErmineUnit):
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
            optimizer = tf.keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        bucket[self.options['DestModel']] = model
