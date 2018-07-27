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

import pickle as pkl
import time
from sample.utils import tcn
from keras.preprocessing.sequence import pad_sequences
from sample.utils.callbacks import ConllevalCallback
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from collections import OrderedDict
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
config.gpu_options.allow_growth = True  # 按需求增长
set_session(tf.Session(config=config))

# Parameters of the network
word_emb_size = 200
chunk_emb_size = 10
dict_emb_size = 15
char_emb_size = 50
cap_emb_size = 5
pos_emb_size = 25
dropout_rate = 0.5  # [0.5, 0.5]

num_classes = 5
epochs = 25
batch_size = 32
max_f = 0
lstm_size = [200]    # BLSTM 隐层大小
learning_rate = 1e-3    # 1e-3  5e-4
decay_rate = learning_rate / epochs     # 1e-6
optimizer = 'rmsprop'    #'rmsprop'
# CNN settings
feature_maps = [25, 25]
kernels = [2, 3]

use_chars = True
use_att = False
batch_normalization = True
highway = False

rootCorpus = r'data'
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
# idx2label = {0: 'O', 1: 'B', 2: 'I'}
idx2label = {0: 'O', 1: 'B-protein', 2: 'I-protein', 3: 'B-gene', 4: 'I-gene'}


print('Loading data...')

with open(rootCorpus + '/train.pkl', "rb") as f:
    train_x, train_y, train_char, train_cap, train_pos, train_chunk, train_dict = pkl.load(f)
with open(rootCorpus + '/test.pkl', "rb") as f:
    test_x, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)
with open(embeddingPath+'/emb.pkl', "rb") as f:
    embedding_matrix = pkl.load(f)
with open(embeddingPath+'/length.pkl', "rb") as f:
    word_maxlen, sentence_maxlen = pkl.load(f)

dataSet = OrderedDict()
dataSet['train'] = [train_x, train_cap, train_pos, train_chunk, train_dict]
dataSet['test'] = [test_x, test_cap, test_pos, test_chunk, test_dict]

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

print(np.asarray(train_x).shape)     # (13697,)
print(np.asarray(train_char).shape)     # (13697, 455, 21)
print(train_y.shape)    # (13697, 455, 3)

print(np.asarray(test_x).shape)     # (4528,)
print(np.asarray(test_char).shape)     # (4528, 455, 21)
print(test_y.shape)    # (4528, 455, 3)

print('done! Model building....')


TIME_STEPS = 21

def createCharDict():
    '''
    创建字符字典
    '''
    import string
    # charSet = set()
    # with open(trainCorpus + '/' + 'train.out', encoding='utf-8') as f:
    #     for line in f:
    #         if not line == '\n':
    #             a = line.strip().split('\t')
    #             charSet.update(a[0])  # 获取字符集合

    char2idx = {}
    char2idx['None'] = len(char2idx)  # 0索引用于填充
    for char in string.printable:
        char2idx[char] = len(char2idx)
    char2idx['**'] = len(char2idx)  # 用于那些未收录的字符
    # print(char2idx)
    return char2idx


if __name__ == '__main__':

    char2idx = createCharDict()
    model, param_str = tcn.dilated_tcn(sentence_maxlen, word_maxlen, char2idx,
                                       char_emb_size, embedding_matrix, TIME_STEPS,
                                       cap_emb_size, pos_emb_size,
                                       chunk_emb_size, dict_emb_size,
                                       num_classes=5,
                                       nb_filters=64,
                                       kernel_size=3,
                                       dilatations=[1, 2, 4],
                                       nb_stacks=6,
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True,)
                                       # output_slice_index='last')
    model.summary()

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
              validation_split=0.2)
              # validation_data=(dataSet['test'], test_y))
    time_diff = time.time() - start_time
    print("%.2f sec for training (4.5)" % time_diff)
    print(model.metrics_names)


    '''
    通过下面的命令启动 TensorBoard
    tensorboard --logdir=/home/administrator/PycharmProjects/keras_bc6_track1/sample/model
    http://localhost:6006
    '''


