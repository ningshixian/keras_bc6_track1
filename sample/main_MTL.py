'''
基于双向LSTM和CRF的药物名实体识别

用训练预料和测试预料一起训练word2vec，使得词向量本身捕捉语义信息

思路：
1、转换为3tag标注问题（0：非实体，1：实体的首词，2：实体的内部词）；
2、获取对应输入的语言学特征（字符特征，词性，chunk，词典特征，大小写）
3、通过双向LSTM，直接对输入序列进行概率预测
4、通过CRF+viterbi算法获得最优标注结果；

%CPU:235%
%RES:12GB
%MEM:18.9%

GPU memory usage: 761M
GPU utils: 47%

'''

import math
import pickle as pkl
import string
import time
from tqdm import tqdm
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import *
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.optimizers import *
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras_contrib.layers import CRF
from keraslayers.ChainCRF import ChainCRF
from keraslayers.ChainCRF import create_custom_objects
import keras.backend as K
import numpy as np
from collections import OrderedDict
from utils.highwayLayer import Highway
from utils.attention_self import Attention_layer
from utils.attention_recurrent import AttentionDecoder

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
config.gpu_options.allow_growth = True  # 按需求增长
set_session(tf.Session(config=config))

# Parameters of the network
char_emb_size = 25
word_emb_size = 50
dropout_rate = 0.5
# dropout_rate = [0.5, 0.5]

epochs = 30
batch_size = 32
maxmax = 1000000

max_f = 0
cdr_max_f = 0
num_corpus = 2

use_chars = True
use_att = False
batch_normalization = False
highway = 0
earlyStopping = 0
lstm_size = [100]    # [100, 75]

optimizer = 'rmsprop'

learning_rate = 1e-3
learning_rate_disc = 2e-4

# CNN settings
feature_maps = [25, 25]
kernels = [2, 3]

# chemCorpus = r'data/chemdner_corpus/'
# cdrCorpus = r'data/cdr_corpus/'
gmCorpus = r'data/BC2GM-IOBES/'
chemdCorpus = r'data/BC4CHEMD-IOBES/'
cdrCorpus = r'data/BC5CDR-IOBES/'
jnlpbaCorpus = r'data/JNLPBA-IOBES/'
ncbiCorpus = r'data/NCBI-disease-IOBES/'

label2idx = {0: 'O', 1: 'B', 2: 'I'}
idx2label_gm = {0: 'O', 1: 'B-GENE', 2: 'I-GENE', 3: 'E-GENE', 4: 'S-GENE'}
idx2label_chemd = {0: 'O', 1: 'B-Chemical', 2: 'I-Chemical', 3: 'E-Chemical', 4: 'S-Chemical'}
idx2label_cdr = {0: 'O', 1: 'B-Chemical', 2: 'I-Chemical', 3: 'E-Chemical', 4: 'S-Chemical',
                 5: 'B-Disease', 6: 'I-Disease', 7: 'E-Disease', 8: 'S-Disease'}
idx2label_jnlpba = {0: 'O', 1: 'B-protein', 2: 'I-protein', 3: 'E-protein', 4: 'S-protein',
                    5: 'B-cell_type', 6: 'I-cell_type', 7: 'E-cell_type', 8: 'S-cell_type',
                    9: 'B-DNA', 10: 'I-DNA', 11: 'E-DNA', 12: 'S-DNA',
                    13: 'B-cell_line', 14: 'I-cell_line', 15: 'E-cell_line', 16: 'S-cell_line',
                    17: 'B-RNA', 18: 'I-RNA', 19: 'E-RNA', 20: 'S-RNA'}
idx2label_ncbi = {0: 'O', 1: 'B-Disease', 2: 'I-Disease', 3: 'E-Disease', 4: 'S-Disease'}

taskList = ['gm', 'chemd', 'cdr', 'jnlpba', 'ncbi']
corpusList = [gmCorpus, chemdCorpus, cdrCorpus, jnlpbaCorpus, ncbiCorpus]
labelList = [idx2label_gm, idx2label_chemd, idx2label_cdr, idx2label_jnlpba, idx2label_ncbi]
main_id = 1
aux_id = 3
num_class_main = len(labelList[main_id])
num_class_aux = len(labelList[aux_id])

print('load data...')

with open(corpusList[main_id] + 'pkl/train.pkl', "rb") as f:
    train_x, train_y, train_char, train_cap = pkl.load(f)
with open(corpusList[main_id] + 'pkl/devel.pkl', "rb") as f:
    devel_x, devel_y, devel_char, devel_cap = pkl.load(f)
with open(corpusList[main_id] + 'pkl/test.pkl', "rb") as f:
    test_x, test_y, test_char, test_cap = pkl.load(f)
with open('embedding/emb.pkl', "rb") as f:
    embedding_matrix, word_maxlen, sentence_maxlen, char2idx = pkl.load(f)

with open(corpusList[aux_id] + 'pkl/train.pkl', "rb") as f:
    cdr_train_x, cdr_train_y, cdr_train_char, cdr_train_cap = pkl.load(f)
with open(corpusList[aux_id] + 'pkl/devel.pkl', "rb") as f:
    cdr_devel_x, cdr_devel_y, cdr_devel_char, cdr_devel_cap = pkl.load(f)
with open(corpusList[aux_id] + 'pkl/test.pkl', "rb") as f:
    cdr_test_x, cdr_test_y, cdr_test_char, cdr_test_cap = pkl.load(f)

print('Data loading completed!')

dataSet = OrderedDict()
dataSet['main'] = [train_x[:maxmax], train_cap[:maxmax]]
dataSet['aux'] = [cdr_train_x[:maxmax], cdr_train_cap[:maxmax]]
dataSet['test'] = [test_x[:maxmax], test_cap[:maxmax]]
dataSet['cdr_test'] = [cdr_test_x[:maxmax], cdr_test_cap[:maxmax]]

print('Data preprocessing....')

# pad the sequences with zero
for key, value in dataSet.items():
    for i in range(len(value)):
        dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

# pad the char sequences with zero list
char_list = [train_char, cdr_train_char, test_char, cdr_test_char]
# char_list = [train_char, cdr_train_char, craft_train_char, test_char, cdr_test_char, craft_test_char]
for item in char_list:
    for j in tqdm(range(len(item))):
        if len(item[j]) < sentence_maxlen:
            item[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(item[j]))))
print(np.array(train_char).shape)   # (61609, 180, 25)

dataSet['main'].insert(1, np.asarray(train_char[:maxmax]))
dataSet['aux'].insert(1, np.asarray(cdr_train_char[:maxmax]))
# dataSet['aux2'].insert(1, np.asarray(craft_train_char))
dataSet['test'].insert(1, np.asarray(test_char[:maxmax]))
dataSet['cdr_test'].insert(1, np.asarray(cdr_test_char[:maxmax]))
# dataSet['craft_test'].insert(1, np.asarray(craft_test_char))


train_y = pad_sequences(train_y[:maxmax], maxlen=sentence_maxlen, padding='post')
cdr_train_y = pad_sequences(cdr_train_y[:maxmax], maxlen=sentence_maxlen, padding='post')
# craft_train_y = pad_sequences(craft_train_y, maxlen=sentence_maxlen, padding='post')

# train_y = to_categorical(train_y, num_classes=3)
# cdr_train_y = to_categorical(cdr_train_y, num_classes=5)
# # craft_train_y = to_categorical(craft_train_y, num_classes=3)

print(train_y.shape)    # (61609, 180, 3)

test_y = pad_sequences(test_y[:maxmax], maxlen=sentence_maxlen, padding='post')
cdr_test_y = pad_sequences(cdr_test_y[:maxmax], maxlen=sentence_maxlen, padding='post')
# craft_test_y = pad_sequences(craft_test_y, maxlen=sentence_maxlen, padding='post')

# test_y = to_categorical(test_y, num_classes=3)
# cdr_test_y = to_categorical(cdr_test_y, num_classes=5)
# # craft_test_y = to_categorical(craft_test_y, num_classes=3)

print(test_y.shape)    # (26453, 180, 3)

y0 = np.ones([len(train_y), 1])
labels0 = to_categorical(y0, num_classes=2)
y1 = np.zeros([len(cdr_train_y), 1])
labels1 = to_categorical(y1, num_classes=2)

y0_t = np.ones([len(test_y), 1])
labels0_t = to_categorical(y0_t, num_classes=2)
y1_t = np.ones([len(cdr_test_y), 1])
labels1_t = to_categorical(y1_t, num_classes=2)

# Add random noise to the labels - important trick!
y0 += 0.05 * np.random.random(y0.shape)
y1 += 0.05 * np.random.random(y1.shape)


def _shared_layer(concat_input):
    '''共享不同任务的Embedding层和bilstm层'''
    cnt = 0
    for size in lstm_size:
        cnt += 1
        if isinstance(dropout_rate, (list, tuple)):
            output = Bidirectional(LSTM(units=size,
                                        return_sequences=True,
                                        dropout=dropout_rate[0],
                                        recurrent_dropout=dropout_rate[1],
                                        # stateful=True, # 上一个batch的最终状态作为下个batch的初始状态
                                        kernel_regularizer=l2(1e-4),
                                        bias_regularizer=l2(1e-4),
                                        implementation=2),
                                   name='shared_varLSTM_' + str(cnt))(concat_input)
        else:
            """ Naive dropout """
            output = Bidirectional(CuDNNLSTM(units=size,
                                             return_sequences=True,
                                             kernel_regularizer=l2(1e-4),
                                             bias_regularizer=l2(1e-4)),
                                   name='shared_LSTM_' + str(cnt))(concat_input)
            if dropout_rate > 0.0:
                output = TimeDistributed(Dropout(dropout_rate), name='shared_drop')(output)
                # output = TimeDistributed(BatchNormalization())(output)

        if use_att:
            # output = Attention_layer()(output)
            output = AttentionDecoder(units=lstm_size[-1],
                             name='attention_decoder_1',
                             output_dim=lstm_size[-1]*2,
                             return_probabilities=False,
                             trainable=True)(output)
    return output


def CNN(seq_length, length, feature_maps, kernels, x):
    '''字符向量学习'''
    concat_input = []
    for filters, size in zip(feature_maps, kernels):
        charsConv1 = TimeDistributed(Conv1D(filters, size, padding='same', activation='relu'))(x)
        charsPool = TimeDistributed(GlobalMaxPool1D())(charsConv1)
        # reduced_l = length - kernel + 1
        # conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        # maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
        concat_input.append(charsPool)

    x = Concatenate()(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x


def buildModel():

    char_embedding = np.zeros([len(char2idx)+1, char_emb_size])
    print(char2idx)
    for key, value in char2idx.items():
        limit = math.sqrt(3.0 / char_emb_size)
        vector = np.random.uniform(-limit, limit, char_emb_size)
        char_embedding[value] = vector

    '''字向量,若为shape=(None,)则代表输入序列是变长序列'''
    tokens_input = Input(shape=(sentence_maxlen,), name='tokens_input', dtype='int32')  # batch_shape=(batch_size,
    tokens_emb = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                           output_dim=embedding_matrix.shape[1],  # 词向量的维度
                           weights=[embedding_matrix],
                           trainable=True,
                           # mask_zero=True,    # 若√则编译报错，CRF不支持？
                           name='token_emd')(tokens_input)

    '''字符向量'''
    chars_input = Input(shape=(sentence_maxlen, word_maxlen,), name='chars_input', dtype='int32')
    chars_emb = TimeDistributed(Embedding(input_dim=char_embedding.shape[0],
                                          output_dim=char_embedding.shape[1],
                                          weights=[char_embedding],
                                          trainable=True,
                                          # mask_zero=True,
                                          name='char_emd'))(chars_input)
    chars_emb = TimeDistributed(Bidirectional(CuDNNLSTM(units=25, return_sequences=False,
                                                        kernel_regularizer=l2(1e-4),
                                                        bias_regularizer=l2(1e-4))))(chars_emb)
    # chars_emb = CNN(sentence_maxlen, word_maxlen, feature_maps, kernels, chars_emb)

    mergeLayers = [tokens_emb, chars_emb]

    # Additional features
    cap_input = Input(shape=(sentence_maxlen,), name='cap_input')
    cap_emb = Embedding(input_dim=2,  # 索引字典大小
                        output_dim=5,  # pos向量的维度
                        trainable=True)(cap_input)
    mergeLayers.append(cap_emb)

    concat_input = concatenate(mergeLayers, axis=-1)  # (none, none, 200)

    if batch_normalization:
        concat_input = BatchNormalization()(concat_input)
    if highway:
        for l in range(highway):
            concat_input = TimeDistributed(Highway(activation='tanh'))(concat_input)

    # Dropout on final input
    concat_input = Dropout(0.5)(concat_input)

    # shared layer
    shared_output = _shared_layer(concat_input)  # (none, none, 200)

    # ======================================================================= #

    # Classifier
    models = {}
    for modelName in ['main', 'aux']:  # , 'aux2'
        output = TimeDistributed(Dense(100, activation=LeakyReLU()))(shared_output)
        if modelName == 'aux':
            output = TimeDistributed(Dense(num_class_aux))(output)
        else:
            output = TimeDistributed(Dense(num_class_main))(output)

        crf = ChainCRF(name=modelName + '_CRF')
        output = crf(output)
        loss_function = crf.sparse_loss

        if optimizer.lower() == 'adam':
            opt = Adam(lr=learning_rate, clipvalue=1., decay=1e-8)
        elif optimizer.lower() == 'nadam':
            opt = Nadam(lr=learning_rate, clipvalue=1.)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(lr=learning_rate, clipvalue=1., decay=1e-8)
        elif optimizer.lower() == 'sgd':
            opt = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=5)

        model = Model(inputs=[tokens_input, chars_input, cap_input], outputs=[output])
        model.compile(loss=loss_function,
                      # loss_weights=[1, 0.05],
                      optimizer=opt,
                      metrics=["accuracy"])
        models[modelName] = model

    # models['discriminator'].summary()
    models['main'].summary()
    return models


if __name__ == '__main__':
    models = buildModel()

    hist_list = []
    from utils.callbacks import ConllevalCallback
    calculatePRF1 = ConllevalCallback(dataSet['test'], test_y, 0, labelList[main_id], sentence_maxlen, flag='main')
    calculatePRF2 = ConllevalCallback(dataSet['cdr_test'], cdr_test_y, 0, labelList[aux_id], sentence_maxlen, flag='aux')

    start_time = time.time()
    for epoch in range(epochs):
        hist = models['main'].fit(x=dataSet['main'], y=[train_y],
                           epochs=1, batch_size=batch_size,
                           shuffle=True,
                           validation_data=(dataSet['test'], [test_y]),
                           callbacks=[calculatePRF1])
        models['aux'].fit(x=dataSet['aux'], y=[cdr_train_y],
                           epochs=1, batch_size=batch_size,
                          shuffle=True,
                           callbacks=[calculatePRF2],
                          validation_data=(dataSet['cdr_test'], [cdr_test_y]))

        hist_list.append(hist.history['val_loss'])

        # # for i in range(5):
        # models['discriminator'].fit(x=dataSet['main'], y=y0, shuffle=True,
        #                    epochs=1, batch_size=batch_size)
        # models['discriminator'].fit(x=dataSet['aux'], y=y1, shuffle=True,
        #                   epochs=1, batch_size=batch_size)
        # # models['discriminator'].fit(x=dataSet['aux2'], y=labels2, shuffle=True,
        # #                             epochs=1, batch_size=batch_size)


    time_diff = time.time() - start_time
    print("%.2f sec for training (4.5)" % time_diff)
    print(models['main'].metrics_names)


    # 看曲线判断模型是否过拟合
    from utils import plot
    plot.plot_val_acc(epochs, hist_list, 'results/val_loss.png')
