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
import time
import pickle as pkl
from keras.layers import *
from keras.models import Model, load_model
from keras.utils import plot_model, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adagrad
from keras.callbacks import Callback
from keras.regularizers import l2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# GPU内存分配
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 按比例
config.gpu_options.allow_growth = True  # 自适应分配
set_session(tf.Session(config=config))


context_window_size = 10
context_network = 'attention'

with open('data/train_cnn.pkl', "rb") as f:
    x_left, x_pos_left, x_right, x_pos_right, y = pkl.load(f)

with open('data/train_cnn2.pkl', "rb") as f:
    x_id, conceptEmbeddings = pkl.load(f)

num_zero = 0
num_ones = 0
for item in y:
    if item==[0]:
        num_zero+=1
    else:
        num_ones+=1
print(num_zero, num_ones)   # 253369 57426

rootCorpus = r'data'
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
with open(embeddingPath+'/emb.pkl', "rb") as f:
    embedding_matrix = pkl.load(f)


def my_dot(inputs):
    a = inputs[0] * inputs[1]
    a = K.sum(a, axis=-1, keepdims=True)
    # a = K.sigmoid(a)
    a = K.softmax(a)
    return a


def to_prob(input):
    sum = K.sum(input, 1, keepdims=True)
    return input / sum


def sum_seq(x):
    return K.sum(x, axis=1, keepdims=False)


def buildAttention3(seq, controller):
    controller_repeated = RepeatVector(context_window_size)(controller)
    attention = merge([controller_repeated, seq], mode='concat', concat_axis=-1)

    attention = Dense(1, activation='softmax')(attention)
    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = merge([attention_repeated, seq], mode='mul')
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

    output_attention_mul = merge(inputs=[seq, attention_repeated], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(output_attention_mul)

    return summed, attention


def buildAttention(seq, controller):

    controller_repeated = RepeatVector(context_window_size)(controller)
    controller_repeated = TimeDistributed(Dense(200))(controller_repeated)

    inputs = [controller_repeated, seq]
    # inputs = merge([controller_repeated, seq], mode='concat', concat_axis=-1)
    # inputs = Dense(200, activation='tanh')(inputs)
    attention = Lambda(my_dot, output_shape=(context_window_size,))(inputs)

    attention = Flatten()(attention)
    attention = Lambda(to_prob, output_shape=(context_window_size,))(attention)

    attention_repeated = RepeatVector(200)(attention)
    attention_repeated = Permute((2, 1))(attention_repeated)

    weighted = merge(inputs=[attention_repeated, seq], mode='mul')
    summed = Lambda(sum_seq, output_shape=(200,))(weighted)

    return summed, attention


def CNN(concat_input):
    shared_layer1 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer2 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    shared_layer3 = Conv1D(200, kernel_size=3, activation='relu', padding='same',
                           kernel_regularizer=l2(1e-4),
                           bias_regularizer=l2(1e-4))
    output = shared_layer1(concat_input)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer2(output)
    output = BatchNormalization(momentum=0.8)(output)
    output = shared_layer3(output)

    output = MaxPooling1D(pool_size=10)(output)   # 加入MaxPooling1D效果降低
    output = Flatten()(output)

    return output


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean



def build_model():
    word_embed_layer = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                                 output_dim=embedding_matrix.shape[1],  # 词向量的维度
                                 weights=[embedding_matrix],
                                 trainable=True,
                                 # mask_zero=True,
                                 name='word_embed_layer')
    candidate_embed_layer = Embedding(input_dim=conceptEmbeddings.shape[0],
                                      output_dim=conceptEmbeddings.shape[1],
                                      input_length=1,
                                      weights=[conceptEmbeddings],
                                      trainable=True)
    pos_embed_layer = Embedding(input_dim=60,  # 索引字典大小
                                output_dim=20,  # pos向量的维度
                                # mask_zero=True,
                                trainable=True)

    # addCandidateInput
    candidate_input = Input(shape=(1,), dtype='int32', name='candidate_input')
    candidate_embed = candidate_embed_layer(candidate_input)
    candidate_flat = Flatten()(candidate_embed)
    controller = candidate_flat
    x_input = [candidate_flat]

    # addContextInput
    left_context_input = Input(shape=(context_window_size,),
                               dtype='int32',
                               name='left_context_input')
    right_context_input = Input(shape=(context_window_size,),
                               dtype='int32',
                               name='right_context_input')
    left_context_embed = word_embed_layer(left_context_input)
    right_context_embed = word_embed_layer(right_context_input)

    left_pos_input = Input(shape=(context_window_size,),
                           dtype='int32',
                           name='left_pos_input')
    right_pos_input = Input(shape=(context_window_size,),
                           dtype='int32',
                           name='right_pos_input')
    left_pos_embed = pos_embed_layer(left_pos_input)
    right_pos_embed = pos_embed_layer(right_pos_input)

    left_rnn_input = [left_context_embed, left_pos_embed]
    left_rnn_input = concatenate(left_rnn_input, axis=-1)
    # left_rnn_input = Dropout(0.5)(left_rnn_input)
    right_rnn_input = [right_context_embed, right_pos_embed]
    right_rnn_input = concatenate(right_rnn_input, axis=-1)
    # right_rnn_input = Dropout(0.5)(right_rnn_input)

    if context_network=='cnn':
        left_rnn = CNN(left_rnn_input)
        right_rnn = CNN(right_rnn_input)
    elif context_network=='gru':
        left_rnn = CuDNNGRU(200, return_sequences=False)(left_rnn_input)
        right_rnn = CuDNNGRU(200, return_sequences=False, go_backwards=True)(right_rnn_input)
    elif context_network == 'attention':
        left_rnn = CuDNNGRU(200, return_sequences=True)(left_rnn_input)
        right_rnn = CuDNNGRU(200, return_sequences=True, go_backwards=True)(right_rnn_input)
        # left_rnn = Conv1D(200, 3, padding='same', activation='relu')(left_rnn_input)
        # right_rnn = Conv1D(200, 3, padding='same', activation='relu')(right_rnn_input)
        left_rnn, attn_values_left = buildAttention(left_rnn, controller)
        right_rnn, attn_values_right = buildAttention(right_rnn, controller)
    x_input.append(left_rnn)
    x_input.append(right_rnn)

    # # S_product
    # tokens_emb = concatenate([left_context_embed, right_context_embed], axis=-1)
    # sim_emb = GlobalAveragePooling1D()(tokens_emb)
    # sim_emb = Multiply()([sim_emb, candidate_embed])
    # sim_emb = Flatten()(sim_emb)

    # # add mention
    # mention_input = Input(shape=(max_mention_words,), dtype='int32', name='mention_input')
    # mention_embed = word_embed_layer(mention_input)
    # mention_mean = Lambda(mask_aware_mean, output_shape=(200,))(mention_embed)
    # x_input.append(mention_mean)

    x_input = concatenate(x_input, axis=-1)
    x_input = Dropout(0.5)(x_input)

    # # 融合上下文和attention结果
    # context = Dense(600)(context)
    # la = Lambda(lambda x: x * 0.5)
    # x_input = Add()([la(x_input), la(context)])

    # build classifier model
    output = Dense(200, activation='relu')(x_input)
    output = Dense(50, activation='relu')(output)
    output = Dense(2, activation='softmax', name='main_output')(output)

    model_input = [candidate_input, left_context_input, right_context_input, left_pos_input, right_pos_input]
    model = Model(inputs=model_input, outputs=[output])

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adagrad = Adagrad(lr=0.002, decay=1e-6, clipvalue=1.)
    model.compile(loss='categorical_crossentropy',   # binary_crossentropy   categorical_crossentropy
                  optimizer='adam',
                  metrics=["accuracy"])

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
            self.model.save('data/weights.hdf5')
            self.max_match = F


if __name__ == '__main__':

    x_id = np.array(x_id)
    x_left = np.array(x_left)
    x_pos_left = np.array(x_pos_left)
    x_right = np.array(x_right)
    x_pos_right = np.array(x_pos_right)
    # x = pad_sequences(x_left, maxlen=sentence_maxlen, padding='post')
    # x_pos = pad_sequences(x_pos, maxlen=sentence_maxlen, padding='post')

    y = to_categorical(y, 2)
    y = np.array(y)

    dataSet = [x_id, x_left, x_right, x_pos_left, x_pos_right]

    print(x_left.shape)  # (367523, 20)
    print(x_id.shape)  # (367523, 1)
    print(x_pos_left.shape)  # (367523, 20)
    print(y.shape)  # (367523, 2)

    filepath = 'data/weights2.{epoch:02d}-{val_acc:.2f}.hdf5'
    saveModel = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                save_best_only=True,  # 只保存在验证集上性能最好的模型
                                save_weights_only=False,
                                mode='auto')
    earlyStop = EarlyStopping(monitor='val_acc', patience=5, mode='auto')

    max_match=0
    call = ConllevalCallback(dataSet, y, max_match)

    start_time = time.time()
    model = build_model()
    model.fit(x=dataSet,
              y=y,
              epochs=20,
              batch_size=32,
              shuffle=True,
              callbacks=[call],
              validation_split=0.2)
    time_diff = time.time() - start_time
    print("%.2f sec for training (%.2f)" % (time_diff, time_diff/60))


    # test
    cnn = load_model('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/weights.hdf5')
    num = 40
    testSet = [x_id[:num], x_left[:num], x_right[:num], x_pos_left[:num], x_pos_right[:num]]
    result = cnn.predict(testSet)
    print(result)

    y2 = []
    for i in range(len(y[:num])):
        y2.append(0 if y[:num][i][0] > y[:num][i][1] else 1)
    print('answer:', y2)

    result2 = []
    for i in range(len(result)):
        result2.append(0 if result[i][0]>result[i][1] else 1)
    print('predict:', result2)

