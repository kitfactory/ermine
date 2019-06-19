import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from ermine import ErmineUnit, OptionInfo, OptionDirection, Bucket


from sklearn import metrics
# confusion_matrix
# from sklearn.metrics import precision_recall_fscore_support

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
            values=['MultiClass','BinaryPickupClass','BinaryPickupClassNegative'])
        
        pickup_class = OptionInfo(
            name='PickupClass',
            direction=OptionDirection.PARAMETER,
            values=['0'])

        pickup_threshold = OptionInfo(
            name='PickupThreshold',
            direction=OptionDirection.PARAMETER,
            values=['0.8'])
    
    
        return [src, data, size, evaluation, pickup_class, pickup_threshold]
    
    def __draw_roc_curve(self, fpr, tpr, auc):
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.show()
        
    def __evaluation_multiple_class_score(self, model, dataset):
        ys = []
        yhats = []
        ids = []
        scores = []
        for x, y, i in dataset:
            score = model.predict(x)[0]
            ids.append(i.numpy())
            yh = score.argmax()
            ys.append(y.numpy().argmax())
        
        confusion =  metrics.confusion_matrix(ys, yhats)
        frame = pd.DataFrame(
            data={'id':ids, 'y':ys, 'yh':yhats, 'score':scores},
            columns=['id','y', 'yh', 'score']
        )
        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/confusion.csv')

    def ___evaluation_binrary_pickup_score(self, model, dataset, pickup_class, pickup_threshold):
        ids = []
        ys = [] # y value
        ybs = [] # y in binary
        yhats = [] # y prediction in binary
        scores = []

        for x, y, i in dataset:
            score = model.predict(x)[0]
            s = score[pickup_class]
            ym = y.numpy().argmax()
            if( s >= pickup_threshold):
                yh = 1
            else:
                yh = 0
            if ym == pickup_class:
                yb = 1
            else:
                yb = 0

            ids.append(i.numpy())
            ys.append(ym)
            ybs.append(yb)
            yhats.append(yh)
            scores.append(score)
        
        frame = pd.DataFrame(
            data={'id':ids, 'y':ys, 'yb':ybs ,'yh':yhats, 'score':scores},
            columns=['id','y', 'yb','yh', 'score']
        )
        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/other_score.csv')
        prfs = metrics.precision_recall_fscore_support(ybs, yhats)
        auc = metrics.auc(fpr, tpr)
        self.__draw_roc_curve(fpr,tpr,auc)

    def ___evaluation_binray_others_score(self, model, dataset,pickup_class,pickup_threshold):
        ids = []
        ys = [] # y value
        ybs = [] # y in binary
        yhats = [] # y prediction in binary
        scores = []

        for x, y, i in dataset:
            score = model.predict(x)[0]
            w_score = score.copy()
            w_score[pickup_class] = 0.0
            s = w_score.max()
            ym = y.numpy().argmax()
            if( s >= pickup_threshold):
                yh = 0
            else:
                yh = 1
            if ym == pickup_class:
                yb = 1
            else:
                yb = 0
            
            ids.append(i.numpy())
            ys.append(ym)
            ybs.append(yb)
            yhats.append(yh)
            scores.append(score)


        frame = pd.DataFrame(
            data={'id':ids, 'y':ys, 'yb':ybs ,'yh':yhats, 'score':scores},
            columns=['id','y', 'yb','yh', 'score']
        )
        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/other_score.csv')
        prfs = metrics.precision_recall_fscore_support(ybs, yhats)
        auc = metrics.auc(fpr, tpr)
        self.__draw_roc_curve(fpr,tpr,auc)


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

        if evaluation == 'MultiClass':
            self.__evaluation_multiple_class_score(model,dataset)
        elif evaluation== 'BinaryPickupClass':
            self.___evaluation_binrary_pickup_score(model,dataset,pickup_class,pickup_threshold)
        else:
            self.___evaluation_binray_others_score(model,dataset,pickup_class,pickup_threshold)

'''

        for x, y, i in dataset:
            score = model.predict(x)[0]
            # print(score[0])
            ids.append(i.numpy())
            yh = -1

            if evaluation == 'OneVersusOne':
                yh = score.argmax()
                ys.append(y.numpy().argmax())
            elif evaluation == 'OneVersusOthers':
                if y == pickup_class:
                    y = 1
                else:
                    y = 0
                s = score[pickup_class]
                if s >= pickup_threshold:
                    yh = 1
                else:
                    yh =  0
                ys.append(y)

            yhats.append(yh)
            scores.append(score)

        confusion =  metrics.confusion_matrix(ys, yhats)
        prfs = metrics.precision_recall_fscore_support(ys, yhats)

        fpr, tpr, thresholds = metrics.roc_curve(ys, yhats)
        auc = metrics.auc(fpr, tpr)

        print(confusion)
        print(prfs)
        self.__draw_roc_curve(fpr,tpr,auc)

'''
    


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
