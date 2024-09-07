from LobTransformer import TransformerBlock
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer, Input, Dense
from tensorflow.keras.models import Model
from keras import backend as K
import numpy as np


class PositionalEncodingLayer(Layer):
    def __init__(self,window_size, **kwargs):
        self.window_size_ = window_size
        super().__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        steps, d_model = x.shape.as_list()[-2:]
        ps = np.zeros([self.window_size_, 1], dtype=K.floatx())
        for tx in range(self.window_size_):
            ps[tx, :] = [(2 / (self.window_size_ - 1)) * tx - 1]

        ps_expand = K.expand_dims(K.constant(ps), axis=0)
        ps_tiled = K.tile(ps_expand, [K.shape(x)[0], 1, 1])

        x = K.concatenate([x, ps_tiled], axis=-1)
        return x


def TransLOB(window_size,n_dim):
    i = Input(shape=(window_size, n_dim))
    x = i
    # 5 Dilated Convolution Layers
    x = tf.keras.layers.Conv1D(42, kernel_size=2, dilation_rate=2, activation='relu', padding='causal')(x)
    x = tf.keras.layers.Conv1D(28, kernel_size=2, dilation_rate=2, activation='relu', padding='causal')(x)
    x = tf.keras.layers.Conv1D(14, kernel_size=2, dilation_rate=8, activation='relu', padding='causal')(x)

    # Layer Norm
    x = tf.keras.layers.LayerNormalization()(x)

    # Positional Encoding
    x = PositionalEncodingLayer(window_size)(x)

    # Transformer Blocks
    #tb1 = TransformerBlock('Block1', 3, True)
    tb2 = TransformerBlock('Block2', 3, True)

    #x = tb1(x)
    x= tb2(x)

    # MLP
    x = tf.keras.layers.Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer='l2', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.1)(x)

    # Output
    out = Dense(3, activation='softmax')(x)

    mdl = Model(inputs=i, outputs=out)
    print(mdl.summary())

    return mdl
