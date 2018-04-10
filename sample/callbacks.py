from sys import stdout
from logging import info
from datetime import datetime
from abc import ABCMeta, abstractmethod
from keras.callbacks import Callback
from decimal import Decimal
from evaluation import conlleval
from evaluation import BIOF1Validation
import numpy as np


class ConllevalCallback(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''

    def __init__(self, X_test, y_test, samples, idx2label, sentence_maxlen, flag):
        super(ConllevalCallback, self).__init__()
        self.X = X_test
        self.y = np.array(y_test)
        self.samples = samples
        self.idx2label = idx2label
        self.sentence_maxlen = sentence_maxlen
        self.flag = flag

    def on_epoch_end(self, epoch, logs={}):
        if self.samples:
            predictions = self.model.predict_generator(self.X, self.samples)
        else:
            print('not train on batch\n')
            predictions = self.model.predict(self.X, verbose=1)

        y_pred = predictions.argmax(axis=-1)  # Predict classes [0]
        y_test = self.y.argmax(axis=-1)

        if self.flag == 'main':
            prf_file = 'results/prf.txt'
            target = r'data/BC4CHEMD-IOBES/test.tsv'
        elif self.flag == 'aux':
            prf_file = 'results/cdr_prf.txt'
            target = r'data/JNLPBA-IOBES/test.tsv'

        pre, rec, f1 = self.predictLabels2(y_pred, y_test)
        p, r, f, c = self.predictLabels1(target, y_pred)

        # if f1 >= self.max_f:
        #     self.max_f = f1
        #     self.model.save('model/Model_best.h5', overwrite=True)
        #     print('do saving')
        # # 预测
        # model = load_model('model/Model_ST.h5', custom_objects=create_custom_objects())

        with open(prf_file, 'a') as pf:
            print('write prf...... ')
            pf.write("epoch= " + str(epoch + 1) + '\n')
            pf.write("precision= " + str(pre) + '\t' + str(p) + '\n')
            pf.write("recall= " + str(rec) + '\t' + str(r) + '\n')
            pf.write("Fscore= " + str(f1) + '\t' + str(f) + '\n')
            pf.write("processed %d tokens with %d phrases;\n" % (c.token_counter, c.found_correct))
            pf.write('found: %d phrases; correct: %d.\n\n' % (c.found_guessed, c.correct_chunk))


    def predictLabels1(self, target, y_pred):
        s = []
        sentences = []
        s_num = 0
        with open(target) as f:
            for line in f:
                if not line == '\n':
                    s.append(line.strip('\n'))
                    continue
                else:
                    # if flag=='main' and not s_num<cal_batch(test_x):
                    #     break
                    # if flag=='aux' and not s_num<cal_batch(cdr_test_x):
                    #     break
                    prediction = y_pred[s_num]
                    s_num += 1
                    for i in range(len(s)):
                        if i >= self.sentence_maxlen: break
                        r = s[i] + '\t' + self.idx2label[prediction[i]] + '\n'
                        sentences.append(r)
                    sentences.append('\n')
                    s = []
        with open('results/result.txt', 'w') as f:
            for line in sentences:
                f.write(str(line))

        p, r, f, c = conlleval.main((None, r'results/result.txt'))
        return round(Decimal(p), 2), round(Decimal(r), 2), round(Decimal(f), 2), c


    def predictLabels2(self, y_pred, y_true):
        # y_true = np.squeeze(y_true, -1)
        lable_pred = list(y_pred)
        lable_true = list(y_true)

        print('\n计算PRF...')
        pre, rec, f1 = BIOF1Validation.compute_f1(lable_pred, lable_true, self.idx2label, 'O', 'IOBES')
        print('precision: {:.2f}%'.format(100. * pre))
        print('recall: {:.2f}%'.format(100. * rec))
        print('f1: {:.2f}%'.format(100. * f1))

        return round(Decimal(100. * pre), 2), round(Decimal(100. * rec), 2), round(Decimal(100. * f1), 2)


class LtlCallback(Callback):
    """Adds after_epoch_end() to Callback.

    after_epoch_end() is invoked after all calls to on_epoch_end() and
    is intended to work around the fixed callback ordering in Keras,
    which can cause output from callbacks to mess up the progress bar
    (related: https://github.com/fchollet/keras/issues/2521).
    """

    def __init__(self):
        super(LtlCallback, self).__init__()
        self.epoch = 0

    def after_epoch_end(self, epoch):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0:
            self.after_epoch_end(self.epoch)
        self.epoch += 1

    def on_train_end(self, logs={}):
        self.after_epoch_end(self.epoch)

class CallbackChain(Callback):
    """Chain of callbacks."""

    def __init__(self, callbacks):
        super(CallbackChain, self).__init__()
        self._callbacks = callbacks

    def _set_params(self, params):
        for callback in self._callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self._callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_end(*args, **kwargs)

class EvaluatorCallback(LtlCallback):
    """Abstract base class for evaluator callbacks."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset, label=None, writer=None):
        super(EvaluatorCallback, self).__init__()
        if label is None:
            label = dataset.name
        if writer is None:
            writer = info
        self.dataset = dataset
        self.label = label
        self.writer = writer
        self.summaries = []

    def __call__(self):
        """Execute Callback. Invoked after end of each epoch."""
        summary = self.evaluation_summary()
        self.summaries.append(summary)
        epoch = len(self.summaries)
        for s in summary.split('\n'):
            self.writer('{} Ep: {} {}'.format(self.label, epoch, s))

    @abstractmethod
    def evaluation_summary(self):
        """Return string summarizing evaluation results."""
        pass

    def after_epoch_end(self, epoch):
        self()

class EpochTimer(LtlCallback):
    """Callback that logs timing information."""

    def __init__(self, label='', writer=info):
        super(EpochTimer, self).__init__()
        self.label = '' if not label else label + ' '
        self.writer = writer

    def on_epoch_begin(self, epoch, logs={}):
        super(EpochTimer, self).on_epoch_begin(epoch, logs)
        self.start_time = datetime.now()

    def after_epoch_end(self, epoch):
        end_time = datetime.now()
        delta = end_time - self.start_time
        start = str(self.start_time).split('.')[0]
        end = str(end_time).split('.')[0]
        self.writer('{}Ep: {} {}s (start {}, end {})'.format(
                self.label, epoch, delta.seconds, start, end
                ))

class Predictor(LtlCallback):
    """Makes and stores predictions for data item sequence."""

    def __init__(self, dataitems):
        super(Predictor, self).__init__()
        self.dataitems = dataitems

    def after_epoch_end(self, epoch):
        predictions = self.model.predict(self.dataitems.inputs)
        self.dataitems.set_predictions(predictions)

class PredictionMapper(LtlCallback):
    """Maps predictions to strings for data item sequence."""

    def __init__(self, dataitems, mapper):
        super(PredictionMapper, self).__init__()
        self.dataitems = dataitems
        self.mapper = mapper

    def after_epoch_end(self, epoch):
        self.dataitems.map_predictions(self.mapper)
        # TODO check if summary() is defined
        info(self.mapper.summary())

class TokenAccuracyEvaluator(EvaluatorCallback):
    """Evaluates performance using token-level accuracy."""

    # TODO why does this class exist? Isn't TokenLevelEvaluator better
    # in every way?

    def __init__(self, dataset, label=None, writer=None):
        super(TokenAccuracyEvaluator, self).__init__(dataset, label, writer)

    def evaluation_summary(self):
        gold = self.dataset.tokens.target_strs
        pred = self.dataset.tokens.prediction_strs
        assert len(gold) == len(pred)
        total = len(gold)
        correct = sum(int(p==g) for p, g in zip(pred, gold))
        return 'acc: {:.2%} ({}/{})'.format(1.*correct/total, correct, total)
