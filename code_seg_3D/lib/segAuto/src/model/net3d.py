# 
# Function to build a 3D dense Neural Network
# B. Sciolla 2017
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.segAuto.src.model import network as nw
import numpy

def tensor_3d_to_compact(x):
    """ Transforms the shape of a tensor from the "3d" convention
        to the "compact"
        Args:
            x (tf.Tensor) --- tensor of shape [case, z, y, x, channel]
        Returns:
            y (tf.Tensor) --- tensor of shape [case_z, y, x, channel]
            nb_cases (int Tensor) --- size of the dimension 'case'
    """
    shape_loc = tf.shape(x)    
    y = tf.reshape( x, [ shape_loc[0] * shape_loc[1],
        shape_loc[2], shape_loc[3], shape_loc[4] ] )   
    
    return y

def tensor_compact_to_3d(x, shape_glob):
    """ Transforms the shape of a tensor from the "compact" convention
        to the "3d"
        Args:
            x (tf.Tensor) --- tensor of shape [case_z, y, x, channel]
            nb_cases (int Tensor) --- size of the dimension 'case'
        Returns:
            y (tf.Tensor) --- tensor of shape [case, z, y, x, channel]
    """
    shape_loc = tf.shape(x)

    y = tf.reshape( x, [ shape_glob[0],
                         shape_glob[1],
                         shape_loc[1],
                         shape_loc[2],
                         shape_loc[3]] )   
    return y

def tensor_compact_to_depth(x, shape_glob):
    """ Transforms the shape of a tensor from the "compact" convention 
        to the "3d"
        Args:
            x (tf.Tensor) --- tensor of shape [case_z, y, x, channel]
            nb_cases (int Tensor) --- size of the dimension 'case'
        Returns:
            y (tf.Tensor) --- tensor of shape [case_y_x, 1, z, channel]
    """
    y = tensor_compact_to_3d(x, shape_glob)
    shape_loc = tf.shape(y)
    
    y = tf.transpose(y, perm = [0,2,3,1,4])
    y = tf.reshape(y, [ 
        shape_glob[0] * shape_glob[2] * shape_glob[3],
        1, shape_glob[1], shape_loc[4] ] )
    return y

def tensor_depth_to_compact(x, shape_glob):
    """ Transforms the shape of a tensor from the "depth" convention 
        to the "compact"
        Args:
            x (tf.Tensor) --- tensor of shape [case_y_x, 1, z, channel]
            nb_cases (int Tensor) --- size of the dimension 'case'
        Returns:
            y (tf.Tensor) --- tensor of shape [case_z, y, x, channel]
    """
    shape_loc = tf.shape(x)
    
        # Go [case_y_x, 1, z, channel] -> [case, y, x, z, channel]
    x = tf.reshape(x, [ 
        shape_glob[0], shape_glob[2], shape_glob[3],
        shape_glob[1], shape_loc[3] ] )
    
        # Go [case, y, x, z, channel] -> [case, z, y, x, channel]
    x = tf.transpose(x, perm = [0,3,1,2,4])
    
        # Go [case, y, x, z, channel] -> [case_z, y, x, channel]
    x = tensor_3d_to_compact(x)
    return x


def testing_fold_unfold():
    """ Unit test function for the 3d - compact - depth transforms
    """
    test = numpy.arange(2*3*4*5*6)
    test = numpy.reshape(test,[2,3,4,5,6])

    x = tf.placeholder(tf.float32,
            [None, None, None, None, None], name='x-input')
    shape_glob = tf.shape(x)

    x1 = tensor_3d_to_compact(x)
    x2 = tensor_compact_to_3d(x1, shape_glob)
    x3 = tensor_compact_to_depth(x1, shape_glob)
    x4 = tensor_depth_to_compact(x3, shape_glob)
    x5 = tensor_compact_to_3d(x4, shape_glob)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    a = sess.run(x2, feed_dict = {x: test})
    b = sess.run(x5, feed_dict = {x: test})

        # Raises an exception if it fails
    numpy.testing.assert_array_almost_equal(a,test)
    numpy.testing.assert_array_almost_equal(b,test)
    
    