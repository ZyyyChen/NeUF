<<<<<<< ours
# 
# Convenience functions for building Residual Neural Networks
# B. Sciolla 2016
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
from lib.segAuto.src.data import convenience as cv

def get_weight_for_cross_entropy(y_in, frequencies, num_classes):
    
    mapprod = tf.to_float(tf.equal(y_in,0))/frequencies[0]
    
    for idxlabel in range(1, num_classes):
        mapprod += tf.to_float(tf.equal(y_in,idxlabel))/frequencies[idxlabel]
        
    return mapprod

def mean_squared_error(pred, y_):
    
    return tf.reduce_mean(tf.square(pred - y_))

'''tf.losses.mean_squared_error(
        y_,
        pred,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES
    )'''

def cross_entropy(out_l, y_l):
   
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_l, logits=out_l)
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_l, logits=out_l)
    
    return tf.reduce_mean(cross_entropy)

def weighted_cross_entropy(out_l, y_l, weights): #frequencies, margins = 0, num_classes = 3, bias_factor_zero_label = 0.1):
    '''weight = \
        cv.weight_frequencies( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
                labels=y_l, logits=out_l) \
                    , y_l, frequencies, num_classes = num_classes, bias_factor_zero_label = bias_factor_zero_label)'''
    
    weights_cross_entropy = tf.constant([weights])
    unweighted_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_l, logits=out_l)#, weights = get_weight_for_cross_entropy(y_l, frequencies, num_classes)
    weighted_cross_entropy = unweighted_cross_entropy * weights_cross_entropy 
    
    '''# Restrict the loss function to a sub-window in 2D
    if margins > 0:
        weighted_cross_entropy = weighted_cross_entropy[:, :, margins:-margins, margins:-margins]'''
        
    return tf.reduce_mean(weighted_cross_entropy)

def get_weights_dice(batchlab, num_classes = 3, epsilon = 1e-10):

    freq = numpy.zeros(num_classes)
    
    for idxclass in range(num_classes):
        freq[idxclass] = numpy.sum(batchlab == idxclass) + 1
        if freq[idxclass] == 0:
            freq[idxclass] = 1e10
            
    fsum = numpy.sum(freq)
    
    weights = numpy.divide(fsum*numpy.ones(num_classes),freq)
    
    #for idxclass in range(num_classes):
    #    if weights[idxclass] == 1:
    #        weights[idxclass] = 0                      
    
    return weights
                                                                    
def get_meandice_cost3d(out_soft, y_, num_classes, margins = 0, background = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
                                                                    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
   
    alldice = tf.compat.v1.div( 1e-10 + 2*XY_sum, 1e-10 + X_sum + Y_sum  )
    # Use log(1-D) as loss per class
    # Average this loss over all classes
    # meandice = tf.reduce_mean(tf.multiply(1 - alldice,[0.1,1,0.5]))
    if background == 0:
        
        if num_classes == 2:
            meandice = 1 - alldice[1]
        else:
            meandice = tf.reduce_mean( 1 - alldice[1:] ) # alldice[1]
            
    else:
        meandice = tf.reduce_mean(1 - alldice)
        
    return alldice,  meandice
                                                                    
def get_meandice_cost(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
        
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients(out_soft,tf.one_hot(y_,num_classes))    
    
    alldice = tf.div( 2*XY_sum, 1e-10 + X_sum + Y_sum  )
    
    # Average this loss over all classes
    meandice = tf.reduce_mean( 1 - alldice )
    
    return alldice, meandice

def get_generalized_cost3d(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
    
    alldice = tf.div( 1e-10 + 2*XY_sum, 1e-10 + X_sum + Y_sum  )   
    meandice = tf.reduce_mean( 1 - alldice ) # alldice[1]  
    
    weights_gendice = get_weights_dice(y_, num_classes)
    weights_gendice = tf.to_float(weights_gendice)
    
    #generaldice = 1 - 2*(weights_gendice[0]*XY_sum[0] + weights_gendice[0]*XY_sum[0] + weights_gendice[2]*XY_sum[2])/(weights_gendice[0]*(X_sum[0]+Y_sum[0]) + weights_gendice[0]*(X_sum[1]+Y_sum[1]) + weights_gendice[2]*(X_sum[2]+Y_sum[2]))
    
    #generaldice = 1 - tf.div(2*tf.reduce_sum(tf.multiply(weights_gendice,XY_sum)),tf.reduce_sum(1e-10 + tf.multiply(weights_gendice,X_sum + Y_sum)))
    
        # Normalisation par cardinal au carre
    #num = tf.multiply(tf.square(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum))),XY_sum)
    #den = tf.multiply(tf.square(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum))), X_sum + Y_sum)
    
        #Normalisation par cardinal de la classe
    num = tf.multiply(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum)),XY_sum)
    den = tf.multiply(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum)), X_sum + Y_sum)
    
    
    generaldice = 1 - 2*tf.reduce_sum(num)/tf.reduce_sum(den)
    
    return alldice, generaldice, meandice

def get_meanjacc_cost3d(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
    
    alldice = tf.div( 2*XY_sum, 1e-10 + X_sum + Y_sum  )   
    meandice = tf.reduce_mean( 1 - alldice ) # alldice[1]
    
    alljacc = tf.div(XY_sum, 1e-10 + X_sum + Y_sum - XY_sum  )  
    meanjacc = tf.reduce_mean( 1 - alljacc)
    
    return alldice, alljacc , meandice, meanjacc

# Make the Dice of the whole Prostate = label 1 and 2 combined
def Dice_pros(y_, out_lay):
    argmax_y = tf.argmax(out_lay,3)
    seg_pros = tf.logical_or(tf.equal(argmax_y,1), tf.equal(argmax_y,2))
    y_pros = tf.logical_or(tf.equal(y_,1), tf.equal(y_,2))
    XYpros_sum, Xpros_sum, Ypros_sum  = Dice_ingredients(seg_pros, y_pros)
    return XYpros_sum, Xpros_sum, Ypros_sum

# Make the Dice of the whole Prostate = label 1 and 2 combined
def Dice_pros3d(y_, out_lay):
    argmax_y = tf.argmax(out_lay,4)
    seg_pros = tf.logical_or(tf.equal(argmax_y,1), tf.equal(argmax_y,2))
    y_pros = tf.logical_or(tf.equal(y_,1), tf.equal(y_,2))
    XYpros_sum, Xpros_sum, Ypros_sum  = Dice_ingredients3d(seg_pros, y_pros)
    return XYpros_sum, Xpros_sum, Ypros_sum
=======
# 
# Convenience functions for building Residual Neural Networks
# B. Sciolla 2016
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
from lib.segAuto.src.data import convenience as cv

def get_weight_for_cross_entropy(y_in, frequencies, num_classes):
    
    mapprod = tf.to_float(tf.equal(y_in,0))/frequencies[0]
    
    for idxlabel in range(1, num_classes):
        mapprod += tf.to_float(tf.equal(y_in,idxlabel))/frequencies[idxlabel]
        
    return mapprod

def mean_squared_error(pred, y_):
    
    return tf.reduce_mean(tf.square(pred - y_))

'''tf.losses.mean_squared_error(
        y_,
        pred,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES
    )'''

def cross_entropy(out_l, y_l):
   
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_l, logits=out_l)
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_l, logits=out_l)
    
    return tf.reduce_mean(cross_entropy)

def weighted_cross_entropy(out_l, y_l, weights): #frequencies, margins = 0, num_classes = 3, bias_factor_zero_label = 0.1):
    '''weight = \
        cv.weight_frequencies( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
                labels=y_l, logits=out_l) \
                    , y_l, frequencies, num_classes = num_classes, bias_factor_zero_label = bias_factor_zero_label)'''
    
    weights_cross_entropy = tf.constant([weights])
    unweighted_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_l, logits=out_l)#, weights = get_weight_for_cross_entropy(y_l, frequencies, num_classes)
    weighted_cross_entropy = unweighted_cross_entropy * weights_cross_entropy 
    
    '''# Restrict the loss function to a sub-window in 2D
    if margins > 0:
        weighted_cross_entropy = weighted_cross_entropy[:, :, margins:-margins, margins:-margins]'''
        
    return tf.reduce_mean(weighted_cross_entropy)

def get_weights_dice(batchlab, num_classes = 3, epsilon = 1e-10):

    freq = numpy.zeros(num_classes)
    
    for idxclass in range(num_classes):
        freq[idxclass] = numpy.sum(batchlab == idxclass) + 1
        if freq[idxclass] == 0:
            freq[idxclass] = 1e10
            
    fsum = numpy.sum(freq)
    
    weights = numpy.divide(fsum*numpy.ones(num_classes),freq)
    
    #for idxclass in range(num_classes):
    #    if weights[idxclass] == 1:
    #        weights[idxclass] = 0                      
    
    return weights
                                                                    
def get_meandice_cost3d(out_soft, y_, num_classes, margins = 0, background = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
                                                                    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
   
    alldice = tf.compat.v1.div( 1e-10 + 2*XY_sum, 1e-10 + X_sum + Y_sum  )
    # Use log(1-D) as loss per class
    # Average this loss over all classes
    # meandice = tf.reduce_mean(tf.multiply(1 - alldice,[0.1,1,0.5]))
    if background == 0:
        
        if num_classes == 2:
            meandice = 1 - alldice[1]
        else:
            meandice = tf.reduce_mean( 1 - alldice[1:] ) # alldice[1]
            
    else:
        meandice = tf.reduce_mean(1 - alldice)
        
    return alldice,  meandice
                                                                    
def get_meandice_cost(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
        
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients(out_soft,tf.one_hot(y_,num_classes))    
    
    alldice = tf.div( 2*XY_sum, 1e-10 + X_sum + Y_sum  )
    
    # Average this loss over all classes
    meandice = tf.reduce_mean( 1 - alldice )
    
    return alldice, meandice

def get_generalized_cost3d(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
    
    alldice = tf.div( 1e-10 + 2*XY_sum, 1e-10 + X_sum + Y_sum  )   
    meandice = tf.reduce_mean( 1 - alldice ) # alldice[1]  
    
    weights_gendice = get_weights_dice(y_, num_classes)
    weights_gendice = tf.to_float(weights_gendice)
    
    #generaldice = 1 - 2*(weights_gendice[0]*XY_sum[0] + weights_gendice[0]*XY_sum[0] + weights_gendice[2]*XY_sum[2])/(weights_gendice[0]*(X_sum[0]+Y_sum[0]) + weights_gendice[0]*(X_sum[1]+Y_sum[1]) + weights_gendice[2]*(X_sum[2]+Y_sum[2]))
    
    #generaldice = 1 - tf.div(2*tf.reduce_sum(tf.multiply(weights_gendice,XY_sum)),tf.reduce_sum(1e-10 + tf.multiply(weights_gendice,X_sum + Y_sum)))
    
        # Normalisation par cardinal au carre
    #num = tf.multiply(tf.square(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum))),XY_sum)
    #den = tf.multiply(tf.square(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum))), X_sum + Y_sum)
    
        #Normalisation par cardinal de la classe
    num = tf.multiply(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum)),XY_sum)
    den = tf.multiply(tf.div(tf.ones([1,num_classes]),tf.maximum(tf.to_float(1),Y_sum)), X_sum + Y_sum)
    
    
    generaldice = 1 - 2*tf.reduce_sum(num)/tf.reduce_sum(den)
    
    return alldice, generaldice, meandice

def get_meanjacc_cost3d(out_soft, y_, num_classes, margins = 0):
    
    ylabels = tf.one_hot(y_, num_classes)
    
    # Restrict with margins
    if margins > 0:
        out_soft = cv.reduce_margins5d(out_soft, margins)
        ylabels  = cv.reduce_margins5d(ylabels, margins)
    
    XY_sum, X_sum, Y_sum = cv.Dice_ingredients3d(out_soft, ylabels)
    
    alldice = tf.div( 2*XY_sum, 1e-10 + X_sum + Y_sum  )   
    meandice = tf.reduce_mean( 1 - alldice ) # alldice[1]
    
    alljacc = tf.div(XY_sum, 1e-10 + X_sum + Y_sum - XY_sum  )  
    meanjacc = tf.reduce_mean( 1 - alljacc)
    
    return alldice, alljacc , meandice, meanjacc

# Make the Dice of the whole Prostate = label 1 and 2 combined
def Dice_pros(y_, out_lay):
    argmax_y = tf.argmax(out_lay,3)
    seg_pros = tf.logical_or(tf.equal(argmax_y,1), tf.equal(argmax_y,2))
    y_pros = tf.logical_or(tf.equal(y_,1), tf.equal(y_,2))
    XYpros_sum, Xpros_sum, Ypros_sum  = Dice_ingredients(seg_pros, y_pros)
    return XYpros_sum, Xpros_sum, Ypros_sum

# Make the Dice of the whole Prostate = label 1 and 2 combined
def Dice_pros3d(y_, out_lay):
    argmax_y = tf.argmax(out_lay,4)
    seg_pros = tf.logical_or(tf.equal(argmax_y,1), tf.equal(argmax_y,2))
    y_pros = tf.logical_or(tf.equal(y_,1), tf.equal(y_,2))
    XYpros_sum, Xpros_sum, Ypros_sum  = Dice_ingredients3d(seg_pros, y_pros)
    return XYpros_sum, Xpros_sum, Ypros_sum
>>>>>>> theirs
