<<<<<<< ours
# ==============================================================================
# Convenience functions for building Residual Neural Networks
# B. Sciolla - M. Martin 2018
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import logging

# 1. Bloquer les messages au niveau du système (C++)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_LEVEL'] = '-1'

# 2. Bloquer les avertissements Python
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')


import tensorflow.compat.v1 as tf
#Désactive spécifiquement les logs de l'API Python
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

from lib.segAuto.src.model import network as nw
from lib.segAuto.src.model import net3d



def UNet_CPPN(params):
    
    x_ = tf.placeholder(tf.float32,[ None, None, None, None, 1], name='x-input') # , None,
    y_ = tf.placeholder(tf.int32,[ None, None, None , None], name='label-output') # [None, None, None, None] 
    keep_prob_ = tf.placeholder(tf.float32, name='dropout_keep_prob')
    shape_glob = tf.shape(x_)
    xp = net3d.tensor_3d_to_compact(x_)
    is_train = tf.placeholder(tf.bool, name = 'is_train')
    
    d = 64
    n_cc = 128
    deconv_size = 2
    batch_norm = True
    batch_norm_pos = 'before_act'
    deconv_activation = 'ReLU'
    deconv_BN = True
    
    num_classes = params['DataParams']['num_classes']
    activation = params['NetworkParams']['activation']
    conv_size = params['NetworkParams']['conv_size'] 
        
    # Floor
    int1 = nw.conv_layer2d(xp, filter_size = [conv_size, conv_size, 1, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int1')
    int2 = nw.conv_layer2d(int1, filter_size = [conv_size, conv_size, d, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int2')
                
    # Stage 1
    int3 = tf.nn.max_pool(int2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int4 = nw.conv_layer2d(int3, filter_size = [conv_size, conv_size, d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int4')    
    int5 = nw.conv_layer2d(int4, filter_size = [conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int5')
    
    # Stage 2
    int6 = tf.nn.max_pool(int5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int7 = nw.conv_layer2d(int6, filter_size = [conv_size, conv_size, 2*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int7')
    int8 = nw.conv_layer2d(int7, filter_size = [conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int8')
    
    # Stage 3
    int9 = tf.nn.max_pool(int8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int10 = nw.conv_layer2d(int9, filter_size = [conv_size, conv_size, 4*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int10')
    int11 = nw.conv_layer2d(int10, filter_size = [conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int11')
   
    # Stage 4
    int12 = tf.nn.max_pool(int11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int13 = nw.conv_layer2d(int12, filter_size = [conv_size, conv_size, 8*d, 16*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int13')
    int14 = nw.conv_layer2d(int13, filter_size = [conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int14')
    
    # Stage 3 bis
    int15 = nw.deconv_layer2d(int14, filter_size = [deconv_size, deconv_size, 8*d, 16*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/8), tf.to_int32(shape_glob[3]/8), 8*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int15')
    int16 = nw.conv_layer2d(tf.concat([int11, int15], axis = 3), filter_size = [conv_size, conv_size, 16*d, 8*d], strides = [1,1,1,1], 
                            padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int16')
    int17 = nw.conv_layer2d(int16, filter_size = [conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int17')
    
    # Stage 2 bis
    int18 = nw.deconv_layer2d(int17, filter_size = [deconv_size, deconv_size, 4*d, 8*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/4), tf.to_int32(shape_glob[3]/4), 4*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int18')
    int19 = nw.conv_layer2d(tf.concat([int8, int18], axis = 3), filter_size = [conv_size, conv_size, 8*d, 4*d], strides = [1,1,1,1], 
                            padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int19')
    int20 = nw.conv_layer2d(int19, filter_size = [conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int20')   
    
    # Stage 1 bis
    int21 = nw.deconv_layer2d(int20, filter_size = [deconv_size, deconv_size, 2*d, 4*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/2), tf.to_int32(shape_glob[3]/2), 2*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int21')
    int22 = nw.conv_layer2d(tf.concat([int5, int21], axis = 3), filter_size = [conv_size, conv_size, 4*d, 2*d], 
                            strides = [1,1,1,1], padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int22')
    int23 = nw.conv_layer2d(int22, filter_size = [conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int23')   
    
    # Floor bis
    int24 = nw.deconv_layer2d(int23, filter_size = [deconv_size, deconv_size, d, 2*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], shape_glob[2], shape_glob[3], d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int24') 
    int25 = nw.conv_layer2d(tf.concat([int2, int24], axis = 3), filter_size = [conv_size, conv_size, 2*d, d], 
                            strides = [1,1,1,1], padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int25')
    int26 = nw.conv_layer2d(int25, filter_size = [conv_size, conv_size, d, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int26')  
    
    # Classification
    int27 = tf.nn.dropout(int26, keep_prob = keep_prob_)
    cl_lay = nw.conv_layer2d(int27, filter_size = [1, 1, d, num_classes], strides = [1,1,1,1], padding = 'SAME', activation = activation,
                             dilations = [1,1,1,1], batch_norm = False, is_train = False, batch_norm_pos = batch_norm_pos, layer_name = 'cl_lay')
    cl_lay= net3d.tensor_compact_to_3d(cl_lay, shape_glob)
    
    # Softmax
    y_soft = tf.nn.softmax(cl_lay)
    
    return x_, keep_prob_, is_train, int1, int2, int3, int4, int5, int6, int7, int8, int9, int10, int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26, cl_lay, y_soft, y_


def VNet_CPPN(params):
    
    x_ = tf.placeholder(tf.float32, [None, None, None, None, 1], name='x-input') # , None,
    y_ = tf.placeholder(tf.int32, [None, None, None, None], name='label-output') # [None, None, None, None] 
    keep_prob_ = tf.placeholder(tf.float32)
    shape_glob = tf.shape(x_)
    is_train = tf.placeholder(tf.bool, name = 'is_train')
    
    d = 16
    deconv_size = 2
    batch_norm = True
    batch_norm_pos = 'before_act'
    deconv_activation = 'ReLU'
    deconv_BN = True
    
    num_classes = params['DataParams']['num_classes']
    activation = params['NetworkParams']['activation']
    conv_size = params['NetworkParams']['conv_size']
          
    # Floor
    int1 = nw.conv_layer3d(x_, filter_size = [conv_size, conv_size, conv_size, 1, d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int1')
    
    int1b = nw.conv_layer3d(int1, [conv_size, conv_size, conv_size, d, d], strides = [1,1,1,1,1], padding = 'SAME', 
                                activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int1b')
        
    int2 = nw.conv_layer3d(int1b + x_, filter_size = [2, 2, 2, d, 2*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation,
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int2')
        
    # Down stage 1
    int3 = nw.conv_layer3d(int2, filter_size = [conv_size, conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int3')
    int4 = nw.conv_layer3d(int3, filter_size = [conv_size, conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int4')
    int5 = nw.conv_layer3d(int4 + int2, filter_size = [2, 2, 2, 2*d, 4*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int5')
    
    # Down stage 2
    int6 = nw.conv_layer3d(int5, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int6')
    int7 = nw.conv_layer3d(int6, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int7')
    int8 = nw.conv_layer3d(int7, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int8')
    int9 = nw.conv_layer3d(int8 + int5, filter_size = [2, 2, 2, 4*d, 8*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int9')
    
    # Down stage 3
    int10 = nw.conv_layer3d(int9, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int10')
    int11 = nw.conv_layer3d(int10, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int11')
    int12 = nw.conv_layer3d(int11, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int12')
    int13 = nw.conv_layer3d(int12 + int9, filter_size = [2, 2, 2, 8*d, 16*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int13')
    
    # Down stage 4
    int14 = nw.conv_layer3d(int13, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int14')
    int15 = nw.conv_layer3d(int14, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int15')
    int16 = nw.conv_layer3d(int15, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int16')
    int17 = nw.deconv_layer3d(int16 + int13, filter_size = [2, 2, 2, 16*d, 16*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/8), tf.to_int32(shape_glob[2]/8), tf.to_int32(shape_glob[3]/8), 16*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int17')
    
    # Up Stage 3
    int18 = nw.conv_layer3d(tf.concat([int12 + int9, int17], axis = 4), filter_size = [conv_size, conv_size, conv_size, 24*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int18')
    int19 = nw.conv_layer3d(int18, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int19')   
    int20 = nw.conv_layer3d(int19, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int20')   
    int21 = nw.deconv_layer3d(int20 + int17, filter_size = [2, 2, 2, 8*d, 16*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/4), tf.to_int32(shape_glob[2]/4), tf.to_int32(shape_glob[3]/4), 8*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int21')
    
    # Up Stage 2
    int22 = nw.conv_layer3d(tf.concat([int8 + int5, int21], axis = 4), filter_size = [conv_size, conv_size, conv_size, 12*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int22')
    int23 = nw.conv_layer3d(int22, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int23')   
    int24 = nw.conv_layer3d(int23, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int24')   
    int25 = nw.deconv_layer3d(int24 + int21, filter_size = [2, 2, 2, 4*d, 8*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/2), tf.to_int32(shape_glob[2]/2), tf.to_int32(shape_glob[3]/2), 4*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int25')
    
    # Up Stage 1
    int26 = nw.conv_layer3d(tf.concat([int4 + int2, int25], axis = 4), filter_size = [conv_size, conv_size, conv_size, 6*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int26')
    int27 = nw.conv_layer3d(int26, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int27')   
    int28 = nw.deconv_layer3d(int27 + int25, filter_size = [2, 2, 2, 2*d, 4*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]), tf.to_int32(shape_glob[2]), tf.to_int32(shape_glob[3]), 2*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int28')
    
    int29 = nw.conv_layer3d(tf.concat([int1b + x_, int28,], axis = 4), filter_size = [conv_size, conv_size, conv_size, 3*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int29')    
    # Classification
    int30 = tf.nn.dropout(int28 + int29, keep_prob_) 
    cl_lay = nw.conv_layer3d(int30, filter_size = [1, 1, 1, 2*d, num_classes], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                             dilations = [1,1,1,1,1], batch_norm = False, is_train = False, batch_norm_pos = batch_norm_pos, layer_name = 'cl_lay')
    
    # Softmax
    y_soft = tf.nn.softmax(cl_lay)
        
    return x_, keep_prob_, is_train, int1, int1b, int2, int3, int4, int5, int6, int7, int8, int9, int10, int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26, int27, int28, int29, int30, cl_lay, y_soft, y_

=======
# ==============================================================================
# Convenience functions for building Residual Neural Networks
# B. Sciolla - M. Martin 2018
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import logging

# 1. Bloquer les messages au niveau du système (C++)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_LEVEL'] = '-1'

# 2. Bloquer les avertissements Python
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')


import tensorflow.compat.v1 as tf
#Désactive spécifiquement les logs de l'API Python
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

from lib.segAuto.src.model import network as nw
from lib.segAuto.src.model import net3d



def UNet_CPPN(params):
    
    x_ = tf.placeholder(tf.float32,[ None, None, None, None, 1], name='x-input') # , None,
    y_ = tf.placeholder(tf.int32,[ None, None, None , None], name='label-output') # [None, None, None, None] 
    keep_prob_ = tf.placeholder(tf.float32, name='dropout_keep_prob')
    shape_glob = tf.shape(x_)
    xp = net3d.tensor_3d_to_compact(x_)
    is_train = tf.placeholder(tf.bool, name = 'is_train')
    
    d = 64
    n_cc = 128
    deconv_size = 2
    batch_norm = True
    batch_norm_pos = 'before_act'
    deconv_activation = 'ReLU'
    deconv_BN = True
    
    num_classes = params['DataParams']['num_classes']
    activation = params['NetworkParams']['activation']
    conv_size = params['NetworkParams']['conv_size'] 
        
    # Floor
    int1 = nw.conv_layer2d(xp, filter_size = [conv_size, conv_size, 1, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int1')
    int2 = nw.conv_layer2d(int1, filter_size = [conv_size, conv_size, d, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int2')
                
    # Stage 1
    int3 = tf.nn.max_pool(int2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int4 = nw.conv_layer2d(int3, filter_size = [conv_size, conv_size, d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int4')    
    int5 = nw.conv_layer2d(int4, filter_size = [conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int5')
    
    # Stage 2
    int6 = tf.nn.max_pool(int5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int7 = nw.conv_layer2d(int6, filter_size = [conv_size, conv_size, 2*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int7')
    int8 = nw.conv_layer2d(int7, filter_size = [conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int8')
    
    # Stage 3
    int9 = tf.nn.max_pool(int8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int10 = nw.conv_layer2d(int9, filter_size = [conv_size, conv_size, 4*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int10')
    int11 = nw.conv_layer2d(int10, filter_size = [conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int11')
   
    # Stage 4
    int12 = tf.nn.max_pool(int11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    int13 = nw.conv_layer2d(int12, filter_size = [conv_size, conv_size, 8*d, 16*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int13')
    int14 = nw.conv_layer2d(int13, filter_size = [conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int14')
    
    # Stage 3 bis
    int15 = nw.deconv_layer2d(int14, filter_size = [deconv_size, deconv_size, 8*d, 16*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/8), tf.to_int32(shape_glob[3]/8), 8*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int15')
    int16 = nw.conv_layer2d(tf.concat([int11, int15], axis = 3), filter_size = [conv_size, conv_size, 16*d, 8*d], strides = [1,1,1,1], 
                            padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int16')
    int17 = nw.conv_layer2d(int16, filter_size = [conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int17')
    
    # Stage 2 bis
    int18 = nw.deconv_layer2d(int17, filter_size = [deconv_size, deconv_size, 4*d, 8*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/4), tf.to_int32(shape_glob[3]/4), 4*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int18')
    int19 = nw.conv_layer2d(tf.concat([int8, int18], axis = 3), filter_size = [conv_size, conv_size, 8*d, 4*d], strides = [1,1,1,1], 
                            padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int19')
    int20 = nw.conv_layer2d(int19, filter_size = [conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int20')   
    
    # Stage 1 bis
    int21 = nw.deconv_layer2d(int20, filter_size = [deconv_size, deconv_size, 2*d, 4*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], tf.to_int32(shape_glob[2]/2), tf.to_int32(shape_glob[3]/2), 2*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int21')
    int22 = nw.conv_layer2d(tf.concat([int5, int21], axis = 3), filter_size = [conv_size, conv_size, 4*d, 2*d], 
                            strides = [1,1,1,1], padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int22')
    int23 = nw.conv_layer2d(int22, filter_size = [conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int23')   
    
    # Floor bis
    int24 = nw.deconv_layer2d(int23, filter_size = [deconv_size, deconv_size, d, 2*d], strides = [1, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0]*shape_glob[1], shape_glob[2], shape_glob[3], d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos,  layer_name = 'int24') 
    int25 = nw.conv_layer2d(tf.concat([int2, int24], axis = 3), filter_size = [conv_size, conv_size, 2*d, d], 
                            strides = [1,1,1,1], padding = 'SAME', activation = activation, dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos,   layer_name = 'int25')
    int26 = nw.conv_layer2d(int25, filter_size = [conv_size, conv_size, d, d], strides = [1,1,1,1], padding = 'SAME', activation = activation, 
                            dilations = [1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int26')  
    
    # Classification
    int27 = tf.nn.dropout(int26, keep_prob = keep_prob_)
    cl_lay = nw.conv_layer2d(int27, filter_size = [1, 1, d, num_classes], strides = [1,1,1,1], padding = 'SAME', activation = activation,
                             dilations = [1,1,1,1], batch_norm = False, is_train = False, batch_norm_pos = batch_norm_pos, layer_name = 'cl_lay')
    cl_lay= net3d.tensor_compact_to_3d(cl_lay, shape_glob)
    
    # Softmax
    y_soft = tf.nn.softmax(cl_lay)
    
    return x_, keep_prob_, is_train, int1, int2, int3, int4, int5, int6, int7, int8, int9, int10, int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26, cl_lay, y_soft, y_


def VNet_CPPN(params):
    
    x_ = tf.placeholder(tf.float32, [None, None, None, None, 1], name='x-input') # , None,
    y_ = tf.placeholder(tf.int32, [None, None, None, None], name='label-output') # [None, None, None, None] 
    keep_prob_ = tf.placeholder(tf.float32)
    shape_glob = tf.shape(x_)
    is_train = tf.placeholder(tf.bool, name = 'is_train')
    
    d = 16
    deconv_size = 2
    batch_norm = True
    batch_norm_pos = 'before_act'
    deconv_activation = 'ReLU'
    deconv_BN = True
    
    num_classes = params['DataParams']['num_classes']
    activation = params['NetworkParams']['activation']
    conv_size = params['NetworkParams']['conv_size']
          
    # Floor
    int1 = nw.conv_layer3d(x_, filter_size = [conv_size, conv_size, conv_size, 1, d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int1')
    
    int1b = nw.conv_layer3d(int1, [conv_size, conv_size, conv_size, d, d], strides = [1,1,1,1,1], padding = 'SAME', 
                                activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int1b')
        
    int2 = nw.conv_layer3d(int1b + x_, filter_size = [2, 2, 2, d, 2*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation,
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int2')
        
    # Down stage 1
    int3 = nw.conv_layer3d(int2, filter_size = [conv_size, conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int3')
    int4 = nw.conv_layer3d(int3, filter_size = [conv_size, conv_size, conv_size, 2*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int4')
    int5 = nw.conv_layer3d(int4 + int2, filter_size = [2, 2, 2, 2*d, 4*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int5')
    
    # Down stage 2
    int6 = nw.conv_layer3d(int5, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int6')
    int7 = nw.conv_layer3d(int6, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int7')
    int8 = nw.conv_layer3d(int7, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int8')
    int9 = nw.conv_layer3d(int8 + int5, filter_size = [2, 2, 2, 4*d, 8*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation, 
                           dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int9')
    
    # Down stage 3
    int10 = nw.conv_layer3d(int9, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int10')
    int11 = nw.conv_layer3d(int10, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int11')
    int12 = nw.conv_layer3d(int11, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int12')
    int13 = nw.conv_layer3d(int12 + int9, filter_size = [2, 2, 2, 8*d, 16*d], strides = [1,2,2,2,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int13')
    
    # Down stage 4
    int14 = nw.conv_layer3d(int13, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int14')
    int15 = nw.conv_layer3d(int14, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int15')
    int16 = nw.conv_layer3d(int15, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int16')
    int17 = nw.deconv_layer3d(int16 + int13, filter_size = [2, 2, 2, 16*d, 16*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/8), tf.to_int32(shape_glob[2]/8), tf.to_int32(shape_glob[3]/8), 16*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int17')
    
    # Up Stage 3
    int18 = nw.conv_layer3d(tf.concat([int12 + int9, int17], axis = 4), filter_size = [conv_size, conv_size, conv_size, 24*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int18')
    int19 = nw.conv_layer3d(int18, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int19')   
    int20 = nw.conv_layer3d(int19, filter_size = [conv_size, conv_size, conv_size, 16*d, 16*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int20')   
    int21 = nw.deconv_layer3d(int20 + int17, filter_size = [2, 2, 2, 8*d, 16*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/4), tf.to_int32(shape_glob[2]/4), tf.to_int32(shape_glob[3]/4), 8*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int21')
    
    # Up Stage 2
    int22 = nw.conv_layer3d(tf.concat([int8 + int5, int21], axis = 4), filter_size = [conv_size, conv_size, conv_size, 12*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int22')
    int23 = nw.conv_layer3d(int22, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int23')   
    int24 = nw.conv_layer3d(int23, filter_size = [conv_size, conv_size, conv_size, 8*d, 8*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int24')   
    int25 = nw.deconv_layer3d(int24 + int21, filter_size = [2, 2, 2, 4*d, 8*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]/2), tf.to_int32(shape_glob[2]/2), tf.to_int32(shape_glob[3]/2), 4*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int25')
    
    # Up Stage 1
    int26 = nw.conv_layer3d(tf.concat([int4 + int2, int25], axis = 4), filter_size = [conv_size, conv_size, conv_size, 6*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int26')
    int27 = nw.conv_layer3d(int26, filter_size = [conv_size, conv_size, conv_size, 4*d, 4*d], strides = [1,1,1,1,1], padding = 'SAME', activation = activation,
                            dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int27')   
    int28 = nw.deconv_layer3d(int27 + int25, filter_size = [2, 2, 2, 2*d, 4*d], strides = [1, 2, 2, 2, 1], 
                           output_shape = tf.stack([shape_glob[0], tf.to_int32(shape_glob[1]), tf.to_int32(shape_glob[2]), tf.to_int32(shape_glob[3]), 2*d]), 
                           padding = 'SAME', activation = deconv_activation, batch_norm = deconv_BN, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int28')
    
    int29 = nw.conv_layer3d(tf.concat([int1b + x_, int28,], axis = 4), filter_size = [conv_size, conv_size, conv_size, 3*d, 2*d], strides = [1,1,1,1,1], padding = 'SAME', 
                            activation = activation, dilations = [1,1,1,1,1], batch_norm = batch_norm, is_train = is_train, batch_norm_pos = batch_norm_pos, layer_name = 'int29')    
    # Classification
    int30 = tf.nn.dropout(int28 + int29, keep_prob_) 
    cl_lay = nw.conv_layer3d(int30, filter_size = [1, 1, 1, 2*d, num_classes], strides = [1,1,1,1,1], padding = 'SAME', activation = activation, 
                             dilations = [1,1,1,1,1], batch_norm = False, is_train = False, batch_norm_pos = batch_norm_pos, layer_name = 'cl_lay')
    
    # Softmax
    y_soft = tf.nn.softmax(cl_lay)
        
    return x_, keep_prob_, is_train, int1, int1b, int2, int3, int4, int5, int6, int7, int8, int9, int10, int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26, int27, int28, int29, int30, cl_lay, y_soft, y_

>>>>>>> theirs
