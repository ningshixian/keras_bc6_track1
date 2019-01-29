'''
实体上下文表示学习c：
1、CNN+attention
2、单层神经网络 f=（词向量平均+拼接+tanh+Dropout）

候选选择：
1、Local modeling 方法
计算所有候选candidate与上下文的相似度，
并对<m,c1>...<m,cx>的得分进行排序 ranking
得分最高者作为mention的id
2、拼接【候选，上下文表示，相似度得分】，softmax分类

组成：
semantic representation layer
convolution layer
pooling layer
concatenation layer (Vm + Vc + Vsim)    Vsim=Vm·M·Vc
hidden layer
softmax layer (0/1)

参考 BMC Bioinformatics
《CNN-based ranking for biomedical entity normalization》
《ACL2018-Linking 》
'''
import os
import time
import csv
import pickle as pkl
from keras.layers import *
from keras.models import Model, load_model
from keras.utils import plot_model, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.callbacks import Callback
from keras.regularizers import l2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import importlib
m=importlib.import_module("sample.4_test_nnet")
import keras.backend as K

# set GPU memory
if 'tensorflow'==K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    # # 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # 按需求增长
    # sess = tf.Session(config=config)
    # # set_session(sess)

    # 方法2:只允许使用x%的显存,其余的放着不动
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5    # 按比例
    sess = tf.Session(config=config)


import sys
print(sys.version)
if sys.version.startswith('3.6'):
    import tensorflow_hub as hub
    # Now instantiate the elmo model
    elmo_model = hub.Module("/home/administrator/PycharmProjects/keras_bc6_track1/sample/temp/moduleA", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())


def my_dot(inputs):
    a = inputs[0] * inputs[1]
    a = K.sum(a, axis=-1, keepdims=True)
    a = K.sigmoid(a)
    # a = K.softmax(a)   # 预测结果全部趋于0，别用softmax?
    return a


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n
    return x_mean


def to_prob(input):
    sum = K.sum(input, 1, keepdims=True)
    return input / sum


def sum_seq(x):
    return K.sum(x, axis=1, keepdims=False)


def max(x):
    x_max = K.max(x, axis=1, keepdims=False)
    return x_max


def buildAttention3(seq, controller):
    controller_repeated = RepeatVector(context_window_size)(controller)
    attention = merge([controller_repeated, seq], mode='concat', concat_axis=-1)

    attention = Dense(1, activation='softmax')(attention)
    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply([attention_repeated, seq])
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)

    return summed, attention


def buildAttention2(seq, controller):
    controller_repeated = RepeatVector(context_window_size)(controller)
    controller = TimeDistributed(Dense(1))(controller_repeated)
    seq1 = TimeDistributed(Dense(1))(seq)
    attention = Add()([controller, seq1])

    attention = Permute((2, 1))(attention)
    # attention = Dense(1, activation='softmax')(attention)
    attention = Dense(context_window_size, activation='softmax')(attention)
    attention = Flatten()(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    output_attention_mul = multiply(inputs=[seq, attention_repeated])
    summed = Lambda(sum_seq, output_shape=(200,))(output_attention_mul)

    return summed, attention


def buildAttention(seq, controller):

    controller_repeated = RepeatVector(context_window_size)(controller)
    controller_repeated = TimeDistributed(Dense(200))(controller_repeated)

    attention = Lambda(my_dot, output_shape=(context_window_size,))([controller_repeated, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)
    return summed, attention


def self_attention(seq):
    # Self-Attention
    attention = Lambda(my_dot, output_shape=(context_window_size,))([seq, seq])

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = multiply(inputs=[attention_repeated, seq])
    # weighted = merge([attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)
    return summed, attention


def CNN(concat_input):
    shared_layer1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer2 = Conv1D(200, kernel_size=4, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer3 = Conv1D(200, kernel_size=5, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    output = shared_layer1(concat_input)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer2(output)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer3(output)

    # output = MaxPooling1D(pool_size=10)(output)   # 加入MaxPooling1D效果降低
    # output = Flatten()(output)

    return output


def hier_CNN(concat_input):
    left_cnn1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(concat_input)
    left_max1 = MaxPooling1D(pool_size=10)(left_cnn1)
    left_max1 = Flatten()(left_max1)
    left_cnn2 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn1)
    left_max2 = MaxPooling1D(pool_size=10)(left_cnn2)
    left_max2 = Flatten()(left_max2)
    left_cnn3 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn2)
    left_max3 = MaxPooling1D(pool_size=10)(left_cnn3)
    left_max3 = Flatten()(left_max3)
    left_cnn4 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(left_cnn3)
    left_max4 = MaxPooling1D(pool_size=10)(left_cnn4)
    left_max4 = Flatten()(left_max4)
    return concatenate([left_max1, left_max2, left_max3, left_max4])


def ElmoEmbedding(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size * [context_window_size])
    },
        signature="tokens",
        as_dict=True)["elmo"]


def build_model():
    word_embed_layer = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                                 output_dim=embedding_matrix.shape[1],  # 词向量的维度
                                 input_length=context_window_size,
                                 weights=[embedding_matrix],
                                 trainable=True,
                                 name='word_embed_layer')

    candidate_embed_layer = Embedding(input_dim=conceptEmbeddings.shape[0],
                                      output_dim=conceptEmbeddings.shape[1],
                                      input_length=1,
                                      weights=[conceptEmbeddings],    # AutoExtend 训练获得
                                      trainable=True,
                                    name='candidate_embed_layer')

    # concept_embed_layer = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
    #                                  output_dim=embedding_matrix.shape[1],  # 词向量的维度
    #                                  input_length=1,
    #                                  weights=[embedding_matrix],
    #                                  trainable=True,
    #                                  name='concept_embed_layer')

    x_input = []
    to_join = []
    attn = []

    # addCandidateInput

    candidate_input = Input(shape=(1,), dtype='int32', name='candidate_input')
    candidate_embed = candidate_embed_layer(candidate_input)
    controller = Flatten()(candidate_embed)

    x_input.append(candidate_input)

    # addContextInput

    left_context_input = Input(shape=(context_window_size,), dtype='int32', name='left_context_input')
    left_context_embed = word_embed_layer(left_context_input)
    # left_elmo_input = Input(shape=(context_window_size,), dtype=K.dtype(K.placeholder(dtype=tf.string)))
    # left_elmo_embedding = Lambda(ElmoEmbedding, output_shape=(context_window_size, 1024))(left_elmo_input)
    # left_context_embed = concatenate([left_context_embed, left_elmo_embedding], axis=-1)
    left_context_embed = Dropout(0.5)(left_context_embed)

    right_context_input = Input(shape=(context_window_size,), dtype='int32', name='right_context_input')
    right_context_embed = word_embed_layer(right_context_input)
    # right_elmo_input = Input(shape=(context_window_size,), dtype=K.dtype(K.placeholder(dtype=tf.string)))
    # right_elmo_embedding = Lambda(ElmoEmbedding, output_shape=(context_window_size, 1024))(right_elmo_input)
    # right_context_embed = concatenate([right_context_embed, right_elmo_embedding], axis=-1)
    right_context_embed = Dropout(0.5)(right_context_embed)

    x_input += [left_context_input, right_context_input]
    # x_input += [left_elmo_input, right_elmo_input]


    if context_network == 'swem_mean':
        left_context = GlobalAveragePooling1D()(left_context_embed)
        right_context = GlobalAveragePooling1D()(right_context_embed)
    elif context_network == 'swem_max':
        left_context = GlobalMaxPooling1D()(left_context_embed)
        right_context = GlobalMaxPooling1D()(right_context_embed)
    elif context_network=='cnn':
        left_cnn = CNN(left_context_embed)
        right_cnn = CNN(right_context_embed)
        left_context, attn_values_left = buildAttention(left_cnn, controller)
        right_context, attn_values_right = buildAttention(right_cnn, controller)
        attn += [attn_values_left, attn_values_right]  # attention 的输出
    elif context_network=='hierarchical convnet':
        left_context = hier_CNN(left_context_embed)
        right_context = hier_CNN(right_context_embed)
    elif context_network=='gru':
        left_context = CuDNNGRU(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(left_context_embed)
        right_context = CuDNNGRU(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(right_context_embed)   # , go_backwards=True
    elif context_network=='lstm':
        left_context = CuDNNLSTM(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(left_context_embed)
        right_context = CuDNNLSTM(200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(right_context_embed)   # , go_backwards=True
    elif context_network=='bigru':
        left_context = Bidirectional(CuDNNGRU(units=200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)))(left_context_embed)
        right_context = Bidirectional(CuDNNGRU(units=200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)))(right_context_embed)
    elif context_network=='bilstm':
        left_context = Bidirectional(CuDNNLSTM(units=200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)))(left_context_embed)
        right_context = Bidirectional(CuDNNLSTM(units=200, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)))(right_context_embed)
    elif context_network == 'bilstm-max':
        left_context = Bidirectional(CuDNNLSTM(units=200,
                                               return_sequences=True,
                                               kernel_regularizer=l2(1e-4),
                                               bias_regularizer=l2(1e-4)))(left_context_embed)
        left_context = GlobalMaxPooling1D()(left_context)
        right_context = Bidirectional(CuDNNLSTM(units=200,
                                                return_sequences=True,
                                                kernel_regularizer=l2(1e-4),
                                                bias_regularizer=l2(1e-4)))(right_context_embed)
        right_context = GlobalMaxPooling1D()(right_context)
    elif context_network == 'bilstm-mean':
        left_context = Bidirectional(CuDNNLSTM(units=200,
                                               return_sequences=True,
                                               kernel_regularizer=l2(1e-4),
                                               bias_regularizer=l2(1e-4)))(left_context_embed)
        left_context = GlobalAveragePooling1D()(left_context)
        right_context = Bidirectional(CuDNNLSTM(units=200,
                                                return_sequences=True,
                                                kernel_regularizer=l2(1e-4),
                                                bias_regularizer=l2(1e-4)))(right_context_embed)
        right_context = GlobalAveragePooling1D()(right_context)

    elif context_network == 'self-attention':
        left_rnn = CuDNNLSTM(200, return_sequences=True, name='gru_l')(left_context_embed)
        right_rnn = CuDNNLSTM(200, return_sequences=True, name='gru_r')(right_context_embed)
        left_context, attn_values_left = self_attention(left_rnn)
        right_context, attn_values_right = self_attention(right_rnn)
        attn += [attn_values_left, attn_values_right]  # attention 的输出
    elif context_network == 'attention':
        left_rnn = CuDNNGRU(200, return_sequences=True, name='gru_l')(left_context_embed)
        right_rnn = CuDNNGRU(200, return_sequences=True, name='gru_r')(right_context_embed)
        left_context, attn_values_left = buildAttention(left_rnn, controller)
        right_context, attn_values_right = buildAttention(right_rnn, controller)
        attn += [attn_values_left, attn_values_right]  # attention 的输出
    elif context_network == 'composition':
        left_rnn = CuDNNGRU(200, return_sequences=True, name='gru_l')(left_context_embed)
        right_rnn = CuDNNGRU(200, return_sequences=True, name='gru_r')(right_context_embed)
        left_context, attn_values_left = buildAttention(left_rnn, controller)
        right_context, attn_values_right = buildAttention(right_rnn, controller)
        # 组合上下文和ID所有相关的信息
        left = [left_context]
        left.append(add([left_context, controller]))
        left.append(subtract([left_context, controller]))
        left.append(multiply([left_context, controller]))

        right = [right_context]
        right.append(add([right_context, controller]))
        right.append(subtract([right_context, controller]))
        right.append(multiply([right_context, controller]))
    else:
        raise ("unknown")

    if mode=='gating':
        # 门控机制
        a = Dense(200, use_bias=False, name='d1')(left_context)
        b = Dense(200, use_bias=False, name='d2')(right_context)
        c = add([a, b])
        # c = Activation('tanh')(c)
        c = Dense(1, activation='sigmoid', use_bias=False, name='d3')(c)
        f = Lambda(lambda x: 1-x)
        left = multiply([c, left_context])
        right = multiply([f(c), right_context])
        context = add([left, right])
        to_join += [context]
        to_join.append(controller)
    elif mode == 'composition':
        to_join += left
        to_join += right
        to_join.append(controller)
    else:
        to_join += [left_context, right_context]
        to_join.append(controller)

    # add mention

    # mention_input = Input(shape=(max_mention_words,), dtype='int32', name='mention_input')
    # mention_embed = word_embed_layer(mention_input)
    # mention_mean = Lambda(mask_aware_mean, output_shape=(200,))(mention_embed)
    # x_input.append(mention_input)
    # to_join.append(mention_mean)

    # join all inputs
    x = concatenate(to_join) if len(to_join) > 1 else to_join[0]

    # build classifier model
    x = Dense(200, activation='relu', name='dense_layer1')(x)
    x = Dense(50, activation='relu', name='dense_layer2')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax', name='main_output')(x)

    model = Model(inputs=x_input, outputs=[output])
    if weights_file is not None:
        model.load_weights(weights_file)

    '''
    loss_function: binary_crossentropy / categorical_crossentropy
    decay=lr/nb_epoch 1e-6
    optimizer: 'adam'/'nadam'/'rmsprop':最后所有样例的输出概率都一样? lr过大
                adagrad: 表现较为正常
                adadelta: 偏向于将样例分类为1
    '''
    lr = 1e-4
    adagrad = Adagrad(lr=lr, decay=lr/10)   # lr=1e-3/5e-3 参数推荐使用默认值 , clipvalue=1.,
    adam = Adam(lr=2e-4, beta_1=0.5)
    rmsprop = RMSprop(lr=1e-4, decay=1e-6)
    sgd = SGD(lr=1e-4, decay=(1e-4)/20, momentum=0.9, nesterov=True)  # 0.001
    model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=["accuracy"])
    attn_model = Model(input=x_input, output=attn)
    print("model compiled!")

    model.summary()
    plot_model(model, to_file='data/model.png', show_shapes=True)
    return model


class ConllevalCallback(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''

    def __init__(self, X_test, y_test, max_match):
        super(ConllevalCallback, self).__init__()
        self.X = X_test
        self.y = y_test
        self.max_match = max_match

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.X)  # 模型预测
        y_pred = predictions.argmax(axis=-1)  # Predict classes [0]
        y_test = self.y.argmax(axis=-1)
        print(predictions[:20])

        TP = 0
        FP = 0
        FN = 0
        for i in range(len(y_test)):
            if y_test[i]==1 and y_pred[i]==1:
                TP+=1
            elif y_test[i]==0 and y_pred[i]==1:
                FP+=1
            elif y_test[i]==1 and y_pred[i]==0:
                FN+=1

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = (2*P*R)/(P+R)

        with open('prf.txt','a') as f:
            f.write('{}\n{}\t{}\t{}'.format(str(epoch), TP,FP,FN))
            f.write('\n')
            f.write('{}\t{}\t{}'.format(P,R,F))
            f.write('\n')

        if F>self.max_match:
            print('\nTP:{}\tF:{}'.format(TP,F))
            # self.model.save('data/weights_arnn'+'.hdf5')
            self.model.save('ned_model/weights_rnn_{}.hdf5'.format(F))
            self.max_match = F


class ConllevalCallback2(Callback):
    '''
    Callback for running the conlleval script on the test dataset after each epoch.
    '''

    def __init__(self, max_match):
        super(ConllevalCallback2, self).__init__()
        self.max_match = max_match

    def on_epoch_end(self, epoch, logs={}):
        for prob in [1, 1.2, 1.5]:   # , float("inf")
            m.main(self.model, prob)
            os.system('python /home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_score.py '
                      '--verbose 1 --force '
                      '/home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_scores '
                      '/home/administrator/桌面/BC6_Track1/test_corpus_20170804/caption_bioc '
                      'system1:/home/administrator/桌面/BC6_Track1/test_corpus_20170804/prediction')
            res = []
            csv_file = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/evaluation/BioID_scorer_1_0_3/scripts/bioid_scores/corpus_scores.csv'
            with open(csv_file) as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                birth_header = next(csv_reader)  # 读取第一行每一列的标题
                i = 0
                for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
                    item = []
                    if i in [2, 8, 14, 20]:
                        res.append(row[14:17])
                    if i == 20:
                        res.append(row[17:])
                        break
                    i += 1
            with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/prf.txt', 'a') as f:
                f.write('{}'.format(epoch))
                f.write('\n')

                for i in range(len(res)):
                    item = res[i]
                    if i == 0:
                        f.write('any, strict: ')
                    if i == 1:
                        f.write('any,overlap: ')
                    if i == 2:
                        f.write('normalized,strict: ')
                    if i == 3:
                        f.write('normalized,overlap: ')
                    f.write('\t'.join(item))
                    f.write('\n')

            F = float(res[-1][2])
            print(F)
            if F>self.max_match:
                self.model.save('ned_model/weights_ned_max.hdf5', overwrite=True)
                self.model.save_weights('ned_model/weights_ned_max.weights', overwrite=True)
                self.max_match = F


if __name__ == '__main__':

    max_match = 0
    context_window_size = 10
    batch_size = 32
    nn = ['hierarchical convnet', 'gru', 'lstm', 'bigru', 'bilstm',
          'bilstm-max', 'bilstm-mean', 'self-attention', 'attention']
    context_network = nn[-1]
    mode = 'gating'
    savedPath = 'data/weights2.{epoch:02d}-{val_ac c:.2f}.hdf5'
    weights_file = None

    rootCorpus = r'data'
    embeddingPath = r'/home/administrator/PycharmProjects/embedding'
    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)

    with open('data/data_train2.pkl', "rb") as f:
        x_left, x_pos_left, x_right, x_pos_right, y, x_elmo_l, x_elmo_r = pkl.load(f)
    # with open('data/data_test2.pkl', "rb") as f:
    #     x_left_test, x_pos_left_test, x_right_test, x_pos_right_test, y_test, x_elmo_l_t, x_elmo_r_t = pkl.load(f)
    with open('data/id_embedding.pkl', "rb") as f:
        x_id, x_id_test, conceptEmbeddings = pkl.load(f)

    print(conceptEmbeddings.shape)  # (12657, 200) 从1开始编号

    num_zero = 0
    num_ones = 0
    for item in y:
        if item == [0]:
            num_zero += 1
        else:
            num_ones += 1
    print(num_zero, num_ones)  # 201631 44518

    # # test
    # x_id_test = np.array(x_id_test + x_id)
    # x_left_test = np.array(x_left_test + x_left)
    # x_right_test = np.array(x_right_test + x_right)
    # y_test = to_categorical(y_test + y, 2)
    # y_test = np.array(y_test)
    # testSet = [x_id_test, x_left_test, x_right_test]

    ind = len(x_id) // batch_size
    # ind = 10

    # train
    x_id = np.array(x_id)
    x_left = np.array(x_left)
    x_pos_left = np.array(x_pos_left)
    x_right = np.array(x_right)
    x_pos_right = np.array(x_pos_right)
    y = to_categorical(y[:batch_size * ind], 2)
    y = np.array(y)
    x_elmo_l = np.array(x_elmo_l)
    x_elmo_r = np.array(x_elmo_r)

    # dataSet = [x_id[:batch_size * ind], x_left[:batch_size * ind], x_right[:batch_size * ind],
    #            x_elmo_l[:batch_size * ind], x_elmo_r[:batch_size * ind]]
    dataSet = [x_id[:batch_size * ind], x_left[:batch_size * ind], x_right[:batch_size * ind]]
    print(x_id.shape, x_left.shape, y.shape)  # (246149, 1) (246149, 10) (246149, 2)

    saveModel = ModelCheckpoint(savedPath,
                                monitor='val_acc',
                                save_best_only=True,  # 只保存在验证集上性能最好的模型
                                save_weights_only=False,
                                mode='auto')
    earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    conllback = ConllevalCallback2(max_match)
    # conllback = ConllevalCallback(dataSet, y, max_match)
    tensorBoard = TensorBoard(log_dir='./ned_model',  # 保存日志文件的地址,该文件将被TensorBoard解析以用于可视化
                              histogram_freq=0)  # 计算各个层激活值直方图的频率(每多少个epoch计算一次)

    start_time = time.time()
    model = build_model()
    model.fit(x=dataSet, y=y,
              epochs=10,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[conllback, tensorBoard, earlyStop],)
              # validation_split=0.1)
    time_diff = time.time() - start_time
    print("Total %.2f min for training" % (time_diff / 60))
    print(max_match)


    # # 加载预训练的模型,再训练
    # root = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/'
    # model = load_model(root + 'ned/ned_model/weights_rnn_0.9412647734274352.hdf5')
    # m.main(model)

