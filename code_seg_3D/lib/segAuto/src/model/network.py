# 
# Convenience functions for building Residual Neural Networks
# B. Sciolla 2016
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy
import random
import numpy.random
import math
from lib.segAuto.src.model import initializer

from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes

_BN_LAYERS = {}
_BN_COUNTER = 0


def _next_bn_name():
    global _BN_COUNTER
    if _BN_COUNTER == 0:
        name = 'batch_normalization'
    else:
        name = 'batch_normalization_' + str(_BN_COUNTER)
    _BN_COUNTER += 1
    return name


def batch_norm_compat(x, is_train, layer_name, momentum=0.9, rank=None, channels=None):
    """Batch norm wrapper compatible with TF 2.21/Keras 3 in TF1-style graphs."""
    if rank == 4 and channels is not None:
        x.set_shape([None, None, None, channels])
    elif rank == 5 and channels is not None:
        x.set_shape([None, None, None, None, channels])

    layer = _BN_LAYERS.get(layer_name)
    if layer is None:
        bn_name = _next_bn_name()
        layer = tf.keras.layers.BatchNormalization(momentum=momentum, name=bn_name)
        if rank is not None and channels is not None:
            layer.build(tuple([None] * (rank - 1) + [channels]))
        _BN_LAYERS[layer_name] = layer

    if isinstance(is_train, bool):
        return layer(x, training=is_train)

    is_train_tensor = tf.reshape(tf.cast(is_train, tf.bool), [])
    return tf.cond(
        is_train_tensor,
        lambda: layer(x, training=True),
        lambda: layer(x, training=False),
    )

def weight_unif_initializer(filter_size, dtype = dtypes.float32, seed = False):
    # Xavier uniform initialization 
    fan_in = filter_size[-2]
    fan_out = filter_size[-1]
    n = (fan_in + fan_out) / 2.0 * filter_size[0] * filter_size[1]
    limit = math.sqrt(3.0 * 1.0 / n)
      
    return random_ops.random_uniform(filter_size, -limit, limit, dtype, seed=seed)

def weight_unif_initializer_3d(filter_size, dtype = dtypes.float32, seed = False):
    # Xavier uniform initialization 
    fan_in = filter_size[-2]
    fan_out = filter_size[-1]
    n = (fan_in + fan_out) / 2.0 * filter_size[0] * filter_size[1]
    limit = math.sqrt(3.0 * 1.0 / n)
      
    return random_ops.random_uniform(filter_size, -limit, limit, dtype, seed=seed)

def moy_deconv_weight_2d(deconv_size, input_size):

    a = tf.constant(1, dtype=tf.float32, shape = [deconv_size, deconv_size, 1, 1])#/(deconv_size*deconv_size)
    
    if input_size != 1:
        b = tf.constant(0, dtype=tf.float32, shape = [deconv_size, deconv_size, input_size-1,1])
        out_init = tf.concat([a,b], axis = 2)
        out_fin = tf.concat([b,a], axis = 2)

        out = out_init

        for i in range(2,input_size):
            b1 = tf.constant(0, dtype=tf.float32, shape = [deconv_size, deconv_size, i-1, 1])
            b2 = tf.constant(0, dtype=tf.float32, shape = [deconv_size, deconv_size, input_size-i, 1])
            out_int = tf.concat([b1,a,b2], axis = 2)
            out = tf.concat([out, out_int], axis = 3)

        out = tf.concat([out, out_fin], axis = 3)
    
    else:
        out = a
    
    return out

def PReLU(x, layer_name):
    alphas = tf.get_variable('alpha_' + layer_name, x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg

def normlu2(x):
    beta = 1.67
    alpha = 5 * beta
    return (1-beta/alpha) * tf.nn.relu(x)  \
        + beta/alpha * tf.nn.relu(x + alpha) - beta

    
def normlu(x):
    beta = 1.67
    return tf.nn.relu(x + beta) - beta


def multipliers_variable(shape):
    """Create a multiplier variable with appropriate initialization."""
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    
    N = 1

    for i in range(len(shape)-1):
        N = N*shape[i]

    stddev = numpy.round(1/numpy.sqrt(N),4)
    
    initial = tf.truncated_normal(shape, stddev = stddev )
    
    return tf.Variable(initial)

def weight_variable_deconv(shape):
    """Create a weight variable with appropriate initialization."""
    
    initial = tf.truncated_normal(shape, mean = 0, stddev = 1 )
    
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv_layer2d(input_tensor, filter_size, strides, padding, activation, dilations, batch_norm, is_train, batch_norm_pos, layer_name):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses elu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    
    with tf.name_scope(layer_name):          
        with tf.name_scope('weights'):
            #weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = tf.contrib.layers.xavier_initializer())
            #weights = tf.Variable(weight_unif_initializer(filter_size), name = 'weights_' + layer_name)
            weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = initializer.xavier_initializer())
             
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[3]])
        with tf.name_scope('conv'):
            out = tf.nn.conv2d(input_tensor, weights, strides, padding, dilations = dilations, name = layer_name ) + biases
                
            # Batch normalization
            if batch_norm and batch_norm_pos == 'before_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_before', momentum=0.9, rank=4, channels=filter_size[3])

            # Activation function    
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)
                
            # Batch normalization
            if batch_norm and batch_norm_pos == 'after_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_after', momentum=0.9, rank=4, channels=filter_size[3])

        return out

    
def conv_layer3d(input_tensor, filter_size, strides, padding, activation, dilations, batch_norm, is_train, batch_norm_pos, layer_name):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses elu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    
    with tf.name_scope(layer_name):          
        with tf.name_scope('weights'):
            #weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = tf.contrib.layers.xavier_initializer())
            weights = tf.Variable(weight_unif_initializer_3d(filter_size), name = 'weights_' + layer_name)
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[4]])
        with tf.name_scope('conv'):
            out = tf.nn.conv3d(input_tensor, weights, strides, padding, dilations = dilations, name = layer_name ) + biases
            
            # Batch normalization
            if batch_norm and batch_norm_pos == 'before_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_before', momentum=0.9, rank=5, channels=filter_size[4])

            # Activation function    
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)
                
            # Batch normalization
            if batch_norm and batch_norm_pos == 'after_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_after', momentum=0.9, rank=5, channels=filter_size[4])

        return out

def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def max_pool3d(x,n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1],
                            strides=[1, n, n, n, 1], padding='SAME')

def deconv2d(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, 
                    strides=[1, stride, stride, 1], padding = 'VALID')

def deconv_layer2d(input_tensor, filter_size, strides, output_shape, padding, activation, batch_norm, is_train, batch_norm_pos, layer_name):
    """Reusable code for a deconv layer of 2x2x2 size

    It does a matrix multiply, bias add, and then uses an elu non-linear function.
    It also sets up name scoping.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      
    # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            #weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = tf.contrib.layers.xavier_initializer(uniform=True))
            #weights = tf.Variable(weight_unif_initializer(filter_size), name = 'weights_' + layer_name) 
            weights = tf.compat.v1.get_variable('weights_' + layer_name, shape = filter_size, initializer = initializer.xavier_initializer())
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[2]])
        with tf.name_scope('conv'):
            out = tf.nn.conv2d_transpose(input_tensor, weights, output_shape, strides, padding, name = layer_name) + biases    
            
            # Batch normalization
            if batch_norm and batch_norm_pos == 'before_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_before', momentum=0.9, rank=4, channels=filter_size[2])

            # Activation function    
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)
                
            # Batch normalization
            if batch_norm and batch_norm_pos == 'after_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_after', momentum=0.9, rank=4, channels=filter_size[2])
            
        return out
    
def moy_deconv_layer2d(input_tensor, filter_size, strides, output_shape, padding, activation, batch_norm, is_train, batch_norm_pos, layer_name):
    """Reusable code for a deconv layer of 2x2x2 size

    It does a matrix multiply, bias add, and then uses an elu non-linear function.
    It also sets up name scoping.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      
    # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = moy_deconv_weight_2d(filter_size[0], filter_size[2])
        with tf.name_scope('conv'):
            out = tf.nn.conv2d_transpose(input_tensor, weights, output_shape, strides, padding, name = layer_name)
            
        return out
    
def deconv_layer3d(input_tensor, filter_size, strides, output_shape, padding, activation, batch_norm, is_train, batch_norm_pos, layer_name):
    """Reusable code for a deconv layer of 2x2x2 size

    It does a matrix multiply, bias add, and then uses an elu non-linear function.
    It also sets up name scoping.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      
    # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            #weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = tf.contrib.layers.xavier_initializer())
            weights = tf.Variable(weight_unif_initializer_3d(filter_size), name = 'weights_' + layer_name)
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[3]])
        with tf.name_scope('conv'):
            out = tf.nn.conv3d_transpose(input_tensor, weights, output_shape, strides, padding, name = layer_name) + biases
            
            # Batch normalization
            if batch_norm and batch_norm_pos == 'before_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_before', momentum=0.9, rank=5, channels=filter_size[3])

            # Activation function    
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)
                
            # Batch normalization
            if batch_norm and batch_norm_pos == 'after_act':
                out = batch_norm_compat(out, is_train, layer_name + '_bn_after', momentum=0.9, rank=5, channels=filter_size[3])
                
    return out

def res_conv_layer2d(input_tensor, filter_size, strides, padding, activation, dilations, nb_outint, batch_norm, is_train, layer_name):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses elu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    outint = input_tensor
    
    for i in range(2, nb_outint+1):
        outint = conv_layer2d(outint, filter_size, strides, padding, activation, dilations, batch_norm = batch_norm, layer_name = layer_name + '_int_' + str(i-1))
        
        # Batch normalization
        if batch_norm:
            outint = batch_norm_compat(outint, is_train, layer_name + '_resbn_' + str(i-1), momentum=0.9, rank=4, channels=filter_size[3])
                
        # Activation function    
        if activation == 'ELU':
            outint = tf.nn.elu(outint)
        elif activation == 'SELU':
            outint = tf.nn.selu(out)
        elif activation == 'PReLU':
            outint = PReLU(outint, layer_name)
        elif activation == 'ReLU':
            outint = tf.nn.relu(outint)
                
    with tf.name_scope(layer_name):          
        with tf.name_scope('weights'):
            weights = tf.compat.v1.get_variable('weights_' + layer_name + '_fin', shape = filter_size, initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[3]])
        with tf.name_scope('conv'):
            out = tf.nn.conv2d(outint, weights, strides, padding, dilations = dilations, name = layer_name ) + biases + input_tensor
            
            # Batch normalization
            if batch_norm:
                out = batch_norm_compat(out, is_train, layer_name + '_resbn_out', momentum=0.9, rank=4, channels=filter_size[3])
                
            # Activation function    
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)

        return out
    
def res_conv_layer3d(input_tensor, filter_size, strides, padding, activation, dilations, nb_outint, batch_norm, layer_name):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses elu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    
    outint = conv_layer3d(input_tensor, filter_size, strides, padding, activation, dilations, layer_name + '_int_1')

    for i in range(2,nb_outint+1):
        outint = conv_layer3d(outint, filter_size, strides, padding, activation, dilations, layer_name + '_int_' + str(i+1))
        
    with tf.name_scope(layer_name):          
        with tf.name_scope('weights'):
            weights = tf.get_variable('weights_' + layer_name, shape = filter_size, initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('biases'):
            biases = bias_variable([filter_size[3]])
        with tf.name_scope('conv'):
            out = tf.nn.conv3d(input_tensor, weights, strides, padding, dilations = dilations, name = layer_name ) + biases + outint
            if activation == 'ELU':
                out = tf.nn.elu(out)
            elif activation == 'SELU':
                out = tf.nn.selu(out)
            elif activation == 'PReLU':
                out = PReLU(out, layer_name)
            elif activation == 'ReLU':
                out = tf.nn.relu(out)

        return out