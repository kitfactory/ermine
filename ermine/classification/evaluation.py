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
        
        prediction_type = OptionInfo(
            name='Prediction',
            direction=OptionDirection.PARAMETER,
            values=['Multiple','Binarization'])
        
        bin_class = OptionInfo(
            name='Prediction.BinarizationClass',
            direction=OptionDirection.PARAMETER,
            values=['0'])

        bin_threshold = OptionInfo(
            name='Prediction.BinarizationThreshold',
            direction=OptionDirection.PARAMETER,
            values=['0.5'])

        return [src, data, size, prediction_type, bin_class, bin_threshold]
    
    def __draw_roc_curve(self, fpr, tpr, auc):
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.show()

    def __draw_confusion_matrix(self,confusion):
        pass
        
    def __predict_multi(self, model, dataset):
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
        confusion.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/multi_confusion.csv')

        frame = pd.DataFrame(
            data={'id':ids, 'y':ys, 'yh':yhats, 'score':scores},
            columns=['id','y', 'yh', 'score']
        )
        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/multi_result.csv')


    def __predict_binary(self, model, dataset,pickup_class, threshold):
        ids = []
        ys = [] # y value
        ybs = [] # y in binary
        yhats = [] # y prediction in binary
        scores = []

        for x, y, i in dataset:
            score = model.predict(x)[0]
            w_score = score.copy()
            w_score[pickup_class] = 0.0
            ps = 1.0 - w_score.max() # positive score
            ym = y.numpy().argmax()
            if( ps >= threshold):
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
        frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/binary_prediction.csv')

        confusion =  metrics.confusion_matrix(ybs, yhats)
        print(confusion)

        # confusion.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/multi_confusion.csv')

        fpr, tpr, thresholds = metrics.roc_curve(ybs, yhats)
        auc = metrics.auc(fpr, tpr)
        roc_curve_frame = pd.DataFrame(
            data = {'fpr':fpr,'tpr':tpr,'threshold':thresholds, 'auc':auc},
            columns = {'fpr','tpr','threshold','auc'}
        )
        roc_curve_frame.to_csv(path_or_buf='/Users/naruhide/Documents/workspace/binary_roc.csv')
        self.__draw_roc_curve(fpr,tpr,auc)

    def run(self, bucket: Bucket):
        f = self.options['ModelFile']
        model: tf.keras.Model = tf.keras.models.load_model(f)
        dataset: tf.data.Dataset = bucket[self.options['TestDataset']]
        size: int = bucket[self.options['TestDatasetSize']]

        evaluation = self.options['Prediction']
        pickup_class = int(self.options['Prediction.BinarizationClass'])
        threshold = float(self.options['Prediction.BinarizationThreshold'])
        dataset = dataset.batch(1)

        if evaluation == 'MultiClass':
            self.__predict_multi(model,dataset)
        elif evaluation == 'Binarization':
            self.__predict_binary(model,dataset,pickup_class,threshold)

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
