import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Lambda, Conv2D
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras.utils import conv_utils


class MaxPoolingWithArgmax2D(Layer):
    '''MaxPooling for unpooling with indices.
    
    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)
    
    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

    def call(self, inputs, **kwargs):
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        padding = self.padding.upper()
        output, argmax = nn_ops.max_pool_with_argmax(inputs, ksize, strides, padding)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]
    
    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]



class MaxUnpooling2D(Layer):
    '''Inversion of MaxPooling with indices.
    
    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)
    
    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
        
        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range
        
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
    
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)
