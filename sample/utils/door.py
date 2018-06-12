from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np
from keras.layers import *

class MyLayer(Layer):

    def __init__(self, dim, **kwargs):
        # self.input_tensor = input_tensor
        # self.char_output_tensor = char_output_tensor
        self.dim = dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # attention_output = concatenate([self.tokens_emb, self.chars_rep], axis=-1)
        # attention_output = TimeDistributed(Dense(200, activation='tanh'))(attention_output)
        # attention_output = TimeDistributed(Dense(200, activation='sigmoid'))(attention_output)
        # left = Multiply()([self.tokens_emb, attention_output])
        # ones = Lambda(lambda x: K.ones_like(x, dtype=None, name=None))(attention_output)
        # right = Multiply()([self.chars_rep, subtract([ones, attention_output])])
        # input_tensor = Add()([left, right])

        input_tensor = x[0]
        char_output_tensor = x[1]
        attention_evidence_tensor = K.concatenate([input_tensor, char_output_tensor], axis=2)
        attention_output = Dense(self.dim, activation='tanh')(attention_evidence_tensor)
        attention_output = Dense(self.dim, activation='sigmoid')(attention_output)
        input_tensor = input_tensor*attention_output + char_output_tensor*(1.0 - attention_output)
        return input_tensor

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])