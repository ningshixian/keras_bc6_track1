'''
基于双向LSTM和CRF的BioID track1

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
from sample.keraslayers.ChainCRF import ChainCRF
from sample.keraslayers.ChainCRF import create_custom_objects
from sample.utils.helpers import createCharDict
from sample.utils.callbacks import ConllevalCallback
import keras.backend as K
import numpy as np
import codecs
from math import ceil
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from collections import OrderedDict

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
config.gpu_options.allow_growth = True  # 按需求增长
set_session(tf.Session(config=config))

# Parameters of the network
word_emb_size = 200
char_emb_size = 25
cap_emb_size = 5
pos_emb_size = 25
chunk_emb_size = 10
dropout_rate = 0.5  # [0.5, 0.5]

num_classes = 3
epochs = 25
batch_size = 32
max_f = 0
highway = 0
lstm_size = [200]    # BLSTM 隐层大小
learning_rate = 1e-3    # 1e-4
decay_rate = learning_rate / epochs     # 1e-6
optimizer = 'rmsprop'    #'rmsprop'
# CNN settings
feature_maps = [25, 25]
kernels = [2, 3]

use_chars = True
use_att = False
batch_normalization = False

rootCorpus = r'data'
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
idx2label = {0: 'O', 1: 'B', 2: 'I'}
# idx2label = {0: 'O', 1: 'B-GENE', 2: 'I-GENE', 3: 'B-PROTEIN', 4: 'I-PROTEIN'}


print('Loading data...')

with codecs.open(rootCorpus + '/train.pkl', "rb") as f:
    train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
with open(rootCorpus + '/test.pkl', "rb") as f:
    test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
with open(embeddingPath+'/emb.pkl', "rb") as f:
    embedding_matrix = pkl.load(f)
with open(embeddingPath+'/length.pkl', "rb") as f:
    word_maxlen, sentence_maxlen = pkl.load(f)

dataSet = OrderedDict()
dataSet['train'] = [train_x, train_cap, train_pos, train_chunk]
dataSet['test'] = [test_x, test_cap, test_pos, test_chunk]

print('done! Preprocessing data....')

# pad the sequences with zero
for key, value in dataSet.items():
    for i in range(len(value)):
        dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

# pad the char sequences with zero list
for j in range(len(train_char)):
    if len(train_char[j]) < sentence_maxlen:
        train_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(train_char[j]))))
for j in range(len(test_char)):
    if len(test_char[j]) < sentence_maxlen:
        test_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(test_char[j]))))

dataSet['train'].insert(1, np.asarray(train_char))
dataSet['test'].insert(1, np.asarray(test_char))

train_y = pad_sequences(train_y, maxlen=sentence_maxlen, padding='post')
test_y = pad_sequences(test_y, maxlen=sentence_maxlen, padding='post')

print(np.asarray(train_char).shape)     # (13697, 418, 23)
print(np.asarray(test_char).shape)     # (4528, 418, 23)
print(train_y.shape)    # (13697, 639, 5)
print(test_y.shape)    # (4528, 639, 5)

print('done! Model building....')


def attention_3d_block(inputs, TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if False:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = concatenate(inputs=[inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


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
            output = Dropout(dropout_rate, name='shared_dropout')(output)

        if use_att:
            attention_mul = attention_3d_block(output, sentence_maxlen)
            output = concatenate([output, attention_mul], axis=-1)
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
    char2idx = createCharDict()

    char_embedding = np.zeros([len(char2idx)+1, char_emb_size])
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
                           # mask_zero=True,    # 若√则编译报错，CuDNNLSTM 不支持？
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
    cap_emb = Embedding(input_dim=5,  # 索引字典大小
                        output_dim=cap_emb_size,  # pos向量的维度
                        trainable=True)(cap_input)
    mergeLayers.append(cap_emb)

    pos_input = Input(shape=(sentence_maxlen,), name='pos_input')
    pos_emb = Embedding(input_dim=60,  # 索引字典大小
                        output_dim=pos_emb_size,  # pos向量的维度
                        trainable=True)(pos_input)
    mergeLayers.append(pos_emb)

    chunk_input = Input(shape=(sentence_maxlen,), name='chunk_input')
    chunk_emb = Embedding(input_dim=25,  # 索引字典大小
                          output_dim=chunk_emb_size,  # chunk 向量的维度
                          trainable=True)(chunk_input)
    mergeLayers.append(chunk_emb)

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

    output = TimeDistributed(Dense(lstm_size[-1], activation='tanh', name='tanh_layer'))(shared_output)
    output = TimeDistributed(Dense(num_classes, kernel_regularizer=l2(1e-4), name='final_layer'))(output)     # 不加激活函数，否则预测结果有问题222222
    crf = ChainCRF(name='CRF')
    output = crf(output)
    loss_function = crf.loss

    if optimizer.lower() == 'adam':
        opt = Adam(lr=learning_rate, clipvalue=1., decay=decay_rate)
    elif optimizer.lower() == 'nadam':
        opt = Nadam(lr=learning_rate, clipvalue=1., decay=decay_rate)
    elif optimizer.lower() == 'rmsprop':
        # opt = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)     # best for LSTM
        opt = RMSprop(lr=learning_rate, clipvalue=1., decay=decay_rate)
    elif optimizer.lower() == 'sgd':
        opt = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
        # opt = SGD(lr=0.001, momentum=0.9, decay=0., nesterov=True, clipvalue=5)

    model = Model(inputs=[tokens_input, chars_input, cap_input, pos_input, chunk_input], outputs=[output])
    model.compile(loss=loss_function,
                  optimizer=opt,
                  metrics=["accuracy"])
    model.summary()

    plot_model(model, to_file='sample/result/model.png', show_shapes=True)
    return model


if __name__ == '__main__':

    # import preprocess
    # preprocess.main()

    model = buildModel()

    calculatePRF1 = ConllevalCallback(dataSet['test'], test_y, 0, idx2label, sentence_maxlen, max_f)
    filepath = 'model/weights1.{epoch:02d}-{val_loss:.2f}.hdf5'
    saveModel = ModelCheckpoint(filepath,
                                monitor='val_loss',
                                save_best_only=True,    # 只保存在验证集上性能最好的模型
                                save_weights_only=False,
                                mode='auto')
    earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    tensorBoard = TensorBoard(log_dir='./model',     # 保存日志文件的地址,该文件将被TensorBoard解析以用于可视化
                              histogram_freq=0)     # 计算各个层激活值直方图的频率(每多少个epoch计算一次)

    start_time = time.time()
    model.fit(x=dataSet['train'], y=train_y,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[calculatePRF1, tensorBoard],
              validation_data=(dataSet['test'], test_y))
    time_diff = time.time() - start_time
    print("%.2f sec for training (4.5)" % time_diff)
    print(model.metrics_names)


    '''
    通过下面的命令启动 TensorBoard
    tensorboard --logdir=/home/administrator/PycharmProjects/keras_bc6_track1/sample/model
    http://localhost:6006
    '''


