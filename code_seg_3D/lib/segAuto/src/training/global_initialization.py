# coding: utf-8

# Resnet
# =====
# 
# Do a resnet for brain data
# 
# 
# General libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pickle
import openpyxl

from lib.segAuto.src.model import network as nw
from lib.segAuto.src.model import neuralnets
from lib.segAuto.src.data import convenience as cv
from lib.segAuto.src.training import cost_functions


sess = None
y_soft = None


def _build_bn_compat_var_map():
    """Map current graph vars to legacy checkpoint keys for BatchNorm layers."""
    var_map = {}
    for var in tf.compat.v1.global_variables():
        graph_name = var.op.name
        if 'batch_normalization' in graph_name:
            ckpt_name = graph_name[graph_name.index('batch_normalization'):]
            var_map[ckpt_name] = var
        else:
            var_map[graph_name] = var
    return var_map


def load_network(dirsave, fileid):
    """ Loading a previously trained network """ 
    if dirsave[-1] != '/':
        dirsave = dirsave + '/'
    saver.restore(sess, dirsave + "test_" + fileid + ".ckpt")
    
def load_best_network(params):
    
    dir_sess = params['TrainingParams']['path_save']
    
    #params['TrainingParams']['results_path'] = "C:/Users/sebastien/Project/current/U&V-net/savenet_/V1_Datasetv2/results/"

    with open(params['TrainingParams']['results_path'] + "train.pickle", 'rb') as pfile :
        try:
            savetrain = pickle.load(pfile)
        except:
            print("No training file in " + dirload)
            
    with open(params['TrainingParams']['results_path'] + "val.pickle", 'rb') as pfile :
        try:
            saveval = pickle.load(pfile)
        except:
            print("No validation file in " + dirload)

    #best_val = str((saveval['loss_list'].index(saveval['loss_best']))*params['TrainingParams']['validation_step'])
    ckpt_path = os.path.join(dir_sess, "test_" + "24500" + ".ckpt")
    try:
        saver.restore(sess, ckpt_path)
    except tf.errors.NotFoundError:
        print("Standard restore failed; retrying with BatchNorm compatibility mapping...")
        compat_saver = tf.compat.v1.train.Saver(var_list=_build_bn_compat_var_map(), max_to_keep=10000)
        compat_saver.restore(sess, ckpt_path)
    
    return savetrain, saveval


def saveparams2xls(params):
    
    if not os.path.isfile(params['DataParams']['filepathsave'] + params['DataParams']['xls_file_name']):
        wb = openpyxl.Workbook()
        wb.create_sheet('params&results', 0)
        
        cells_name = ['Flip', 'Affine', 'Gaussian random fields', 'Gaussian white noise', 'rotate', 
                      'any', 'Patch size', 'Patch/batch', 'Conv size', 'Activation', 
                      'loss', 'weight_cross_entropy', 'Nb training case', 'learning_rate', 'nb_steps', 
                      'Step at convergence', 'best validation loss', 'DICE Training', 'DICE validation', 
                      'Dice Test', 'Hausdorff test', 'MAD test', 'Seg time', 'patch_depth 3D', 
                      'recouv', 'nb_params', 'Runtime']
        
        column = ['D', 'E', 'F', 'G', 'H',
                    'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W',
                    'X', 'Y', 'Z', 'AA', 'AB',
                    'AC', 'AD', 'AE']
        
        wb['params&results']['B1'] = 'Session'
        wb['params&results']['C1'] = 'Network'
        wb['params&results']['D1'] = 'Data augmentation'
        wb['params&results']['J1'] = 'Training parameters'
        wb['params&results']['S1'] = 'Results'       
        
        for i in range(len(cells_name)):
            wb['params&results'][column[i] + '2'] = cells_name[i]
            
    else:
        wb = openpyxl.load_workbook(params['DataParams']['filepathsave'] + params['DataParams']['xls_file_name'])

    i = 2
    test = []
    while test != None:
        i += 1
        test = wb['params&results']['B' + str(i)].value
           
    wb['params&results']['B' + str(i)] = "'" + params['TrainingParams']['path_save'][-22:-1] + "'" #params['TrainingParams']['path_save'][-31:]
    wb['params&results']['C' + str(i)] = params['NetworkParams']['Network']
    
    if any(params['DataParams']['data_aug'].values()):
        wb['params&results']['D' + str(i)] = str(params['DataParams']['data_aug']['fliplr']) + ' (' + str(params['DataParams']['data_aug_prob']['prob_flip']) + ')' 
        wb['params&results']['E' + str(i)] = str(params['DataParams']['data_aug']['affine']) + ' (' + str(params['DataParams']['data_aug_prob']['prob_affine']) + ')'
        wb['params&results']['F' + str(i)] = str(params['DataParams']['data_aug']['gaussian_random_fields']) + ' (' + str(params['DataParams']['data_aug_prob']['prob_grf']) + ')'
        wb['params&results']['G' + str(i)] = str(params['DataParams']['data_aug']['gaussian_white_noise']) + ' (' + str(params['DataParams']['data_aug_prob']['prob_gwn']) + ')'
        wb['params&results']['H' + str(i)] = str(params['DataParams']['data_aug']['rotate']) + ' (' + str(params['DataParams']['data_aug_prob']['prob_rotate']) + ')'
        wb['params&results']['I' + str(i)] = str(params['DataParams']['data_aug_prob']['prob_anytransform'])
        
    wb['params&results']['J' + str(i)] = str(params['DataParams']['patch_size'])
    wb['params&results']['K' + str(i)] = params['DataParams']['training_batchsize'] 
    wb['params&results']['L' + str(i)] = params['NetworkParams']['conv_size']    
    wb['params&results']['M' + str(i)] = params['NetworkParams']['activation']
    wb['params&results']['N' + str(i)] = params['NetworkParams']['Loss']
    
    if params['NetworkParams']['Loss'] in [ 'weighted_cross_entropy', 'weighted_cross_entropy_softDice']:
        wb['params&results']['O' + str(i)] = str(params['NetworkParams']['cross_entropy_weight'])
        
    wb['params&results']['P' + str(i)] = len(params['DataParams']['learning_cases'])
    wb['params&results']['Q' + str(i)] = str(params['TrainingParams']['learning_rates'])
    wb['params&results']['R' + str(i)] = str(params['TrainingParams']['nb_steps'])
        
    wb.save(params['DataParams']['filepathsave'] + params['DataParams']['xls_file_name'])
    
    print('Enregistrement des parametres dans ' + params['DataParams']['filepathsave'] + params['DataParams']['xls_file_name'] + ' terminé')
    
    return

def initialize_network(params):
   
    # Réinitialisation du graphe par défault
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    # Clear batch norm layers cache when reinitializing graphs
    try:
        nw._BN_LAYERS.clear()
        nw._BN_COUNTER = 0
    except Exception:
        pass
    
    # Define network 
    if params['NetworkParams']['Network'] == 'UNet_CPPN' :
        x_, keep_prob_, is_train, int1, int2, int3, int4, int5, int6, int7, int8, int9,\
        int10, int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26,\
        cl_lay, y_soft, y_ = neuralnets.UNet_CPPN(params)
           
    elif params['NetworkParams']['Network'] == 'VNet_CPPN':    
        x_, keep_prob_, is_train, int1, int1b, int2, int3, int4, int5, int6, int7, int8, int9, int10,\
        int11, int12, int13, int14, int15, int16, int17, int18, int19, int20, int21, int22, int23, int24, int25, int26, int27,\
        int28, int29, int30, cl_lay, y_soft, y_ = neuralnets.VNet_CPPN(params)
        
    # Build some training parameters
    frequencies_ = tf.compat.v1.placeholder(tf.float32, shape=[params['DataParams']['num_classes']])
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
    weight_dice = tf.compat.v1.placeholder(tf.float32, shape=[])
    
    # Build convergence and performance indicators  
    accuracy, XY_sum, X_sum, Y_sum = cv.compute_all_validation3d(y_, y_soft, params['DataParams']['num_classes']) #
    
    # Losses   
    weighted_cross_entropy = cost_functions.weighted_cross_entropy(cl_lay, y_, params['NetworkParams']['cross_entropy_weight'])
    
    cross_entropy = cost_functions.cross_entropy(cl_lay, y_)
    
    alldice,  meandice = cost_functions.get_meandice_cost3d(y_soft, y_, params['DataParams']['num_classes'], 0, 0)
    
    if params['NetworkParams']['Loss'] == 'weighted_cross_entropy_softDice':
        loss = (1-weight_dice)*weighted_cross_entropy + weight_dice*meandice
    
    elif params['NetworkParams']['Loss'] == 'cross_entropy_softDice':
        loss = (1-weight_dice)*cross_entropy + weight_dice*meandice
        
    elif params['NetworkParams']['Loss'] == 'cross_entropy':
        loss = cross_entropy   
        
    elif params['NetworkParams']['Loss'] == 'softDice':
        loss = meandice
    
    elif params['NetworkParams']['Loss'] == 'weighted_cross_entropy':
        loss = weighted_cross_entropy

    elif params['NetworkParams']['Loss'] == 'Dice_and_MAD':
        loss = meandice
        
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # Création d'une nouvelle session
    sess = tf.compat.v1.Session()
    
    # Enregistrement du graph
    logdir = os.path.join(params['TrainingParams']['results_path'], 'logdir')
    if os.path.exists(logdir) and not os.path.isdir(logdir):
        print("Warning: logdir path exists and is not a directory. Removing and recreating.")
        os.remove(logdir)
    os.makedirs(logdir, exist_ok=True)

    # TensorFlow gfile/FileWriter may fail on paths containing non-ASCII chars (Windows).
    logdir_abspath = os.path.abspath(logdir)
    try:
        logdir_abspath.encode('ascii')
        safe_logdir = logdir_abspath
    except UnicodeEncodeError:
        safe_logdir = os.path.join(os.getcwd(), 'logdir')
        print(f"Warning: TensorFlow summary path contains non-ASCII characters; writing TF logs to fallback path: {safe_logdir}")
        os.makedirs(safe_logdir, exist_ok=True)

    writer = tf.compat.v1.summary.FileWriter(safe_logdir, sess.graph)
    writer.close()
    
    # Initilisation des variables 
    sess.run(tf.compat.v1.global_variables_initializer()) # Graphe variable initialization
    
    saver = tf.compat.v1.train.Saver(max_to_keep = 10000)   
    
    # Make all local variables global (!)
    globals().update(locals())