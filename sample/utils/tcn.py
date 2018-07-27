import math
from keras.layers import *
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from sample.utils.helpers import createCharDict
import keras.backend as K
import numpy as np
from sample.keraslayers.ChainCRF import ChainCRF
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
config.gpu_options.allow_growth = True  # 按需求增长
set_session(tf.Session(config=config))


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])


def residual_block(x, s, i, activation, nb_filters, kernel_size):
    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='causal',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(0.05)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = add([original_x, x])
    return res_x, x


def dilated_tcn(sentence_maxlen, word_maxlen,
                char2idx, char_emb_size, embedding_matrix, TIME_STEPS,
                cap_emb_size, pos_emb_size,
                chunk_emb_size, dict_emb_size,
                num_classes, nb_filters,
                kernel_size, dilatations, nb_stacks,
                activation='wavenet', use_skip_connections=True,
                return_param_str=False, output_slice_index=None,
                regression=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """
    char_embedding = np.zeros([len(char2idx) + 1, char_emb_size])
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
    chars_lstm_out = TimeDistributed(Bidirectional(CuDNNLSTM(units=char_emb_size, return_sequences=True,
                                                             kernel_regularizer=l2(1e-4),
                                                             bias_regularizer=l2(1e-4))))(chars_emb)
    # Character-level model
    chars_attention = TimeDistributed(Permute((2, 1)))(chars_lstm_out)
    chars_attention = TimeDistributed(Dense(TIME_STEPS, activation='softmax'))(chars_attention)
    chars_attention = TimeDistributed(Permute((2, 1), name='attention_vec'))(chars_attention)
    chars_attention = Multiply()([chars_lstm_out, chars_attention])
    chars_attention = TimeDistributed(GlobalAveragePooling1D())(chars_attention)

    chars_lstm_final = TimeDistributed(Bidirectional(CuDNNLSTM(units=char_emb_size, return_sequences=False,
                                                               kernel_regularizer=l2(1e-4),
                                                               bias_regularizer=l2(1e-4)),
                                                     merge_mode='concat'))(chars_emb)
    # chars_emb = CNN(sentence_maxlen, word_maxlen, feature_maps, kernels, chars_emb)
    chars_rep = Concatenate(axis=-1)([chars_attention, chars_lstm_final])

    mergeLayers = [tokens_emb, chars_rep]

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

    dict_input = Input(shape=(sentence_maxlen,), name='dict_input')
    dict_emb = Embedding(input_dim=5,
                          output_dim=dict_emb_size,  # dict 向量的维度
                          trainable=True)(dict_input)
    mergeLayers.append(dict_emb)

    concat_input = concatenate(mergeLayers, axis=-1)  # (none, none, 200)



    x = concat_input
    x = Convolution1D(nb_filters, kernel_size, padding='causal', name='initial_conv')(x)

    skip_connections = []
    for s in range(nb_stacks):
        for i in dilatations:
            x, skip_out = residual_block(x, s, i, activation, nb_filters, kernel_size)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = add(skip_connections)
    x = Activation('relu')(x)

    if output_slice_index is not None:  # can test with 0 or -1.
        if output_slice_index == 'last':
            output_slice_index = -1
        if output_slice_index == 'first':
            output_slice_index = 0
        x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)

    print('x.shape=', x.shape)

    # classification
    # x = Dense(num_classes)(x)
    # x = Activation('softmax', name='output_softmax')(x)
    x = TimeDistributed(Dense(64, activation='tanh', name='tanh_layer'))(x)
    x = TimeDistributed(Dense(num_classes, name='final_layer'))(x)
    crf = ChainCRF(name='CRF')
    x = crf(x)

    output_layer = x
    print('model.x = {}'.format(concat_input.shape))    # (?, 455, 455)
    print('model.y = {}'.format(output_layer.shape))    # (?, 455, 5)
    model_input = [tokens_input, chars_input, cap_input, pos_input, chunk_input, dict_input]
    model = Model(model_input, output_layer)

    adam = Adam(lr=0.002, clipnorm=1.)
    rmsprop = RMSprop(lr=0.002, clipnorm=1.0)
    # model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(adam, loss=crf.loss, metrics=['accuracy'])
    print('Adam with norm clipping.')

    if return_param_str:
        param_str = 'D-TCN_C{}_B{}_L{}'.format(2, nb_stacks, dilatations)
        return model, param_str
    else:
        return model