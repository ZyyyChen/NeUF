<<<<<<< ours
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import os
import shutil
import time
import pickle
from src.data import dataset
import scipy.io as sio
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_database_matfile(params):
    """ Create dataset from several matlab file """
    
    filepathname = params['DataParams']['filepathname']
    data_list = params['DataParams']['data_list']
    
    images = []
    labels = []

    print('Loaded data :\n')
    
    for i in range(len(data_list)):
        
        data = sio.loadmat(filepathname + data_list[i] + '.mat')  
        data['volume'] = data['volume'].astype(np.uint8)   
        data['label'] = data['label'].astype(np.uint8)
        
        data_temp = data['volume']
           
        if params['NetworkParams']['Network'] in  ['UNet_CPPN']:
            
            data_norm = np.zeros([data_temp.shape[0], data_temp.shape[1], data_temp.shape[2]])
            for j in range(data_norm.shape[0]):
                if data_temp[j,:,:].std() != 0:
                    data_norm[j,:,:] = (data_temp[j,:,:] - data_temp[j,:,:].mean())/data_temp[j,:,:].std()

        elif params['NetworkParams']['Network'] in ['VNet_CPPN']:
        
            data_norm = (data_temp - data_temp.mean())/data_temp.std()
        
        images.append(data_norm)
        labels.append(data['label'])
        
        if (os.path.isfile(filepathname + data_list[i] + '_image.nii') or os.path.isfile(filepathname + data_list[i] + '_label.nii')) == False:
            
            image_loc = nib.Nifti1Image(np.int16(np.transpose(data['volume'], (2,0,1))), -np.eye(4))
            #nib.save(image_loc, filepathname + data_list[i] + '_image.nii')
            
            for j in range(max(data['label'].reshape(-1))):
                
                label_loc = data['label'] == j + 1
                label_loc = nib.Nifti1Image(np.int16(np.transpose(label_loc, (2,0,1))), -np.eye(4))
                
                #nib.save(label_loc, filepathname + data_list[i] + '_label_' + str(j + 1) + '.nii')
            
            #del image_loc, label_loc

        del data, data_temp, data_norm 
        
        print(filepathname + data_list[i])
        
    return images, labels



def read_data_set(init_params):
    """ The exposed read function. Fetch the directory in the file dirfile.txt
        tagstr : a string containing the tag of the dataset to load (default, '1')
    """
    
    params = copy.copy(init_params)
    
# Load dataset

    #lowres_data, highres_data, label = load_one_matfile(params['DataParams']['filepathname'])
    images, labels = load_database_matfile(params)

    
# Creation of a dictionnary which contains all the network parameters 
    
    # Set parameters from loaded data
    params['DataParams']['num_volumes'] = len(images)    
    params['DataParams']['patchxin'] = params['DataParams']['patch_size'][1] + params['DataParams']['augx'] * 2
    params['DataParams']['patchyin'] = params['DataParams']['patch_size'][2] + params['DataParams']['augx'] * 2
    params['DataParams']['patchdepth'] = params['DataParams']['patch_size'][0]
    params['DataParams']['patchxout'] = params['DataParams']['patch_size'][1] 
    params['DataParams']['patchyout'] = params['DataParams']['patch_size'][2]
    params['DataParams']['num_classes'] = np.asarray(labels).reshape(-1).max() + 1 
    
    # Definition of the gradient step and Dice loss start
    learning_rate_list = np.zeros(np.sum(params['TrainingParams']['nb_steps']))
    weight_dice_list = np.zeros(np.sum(params['TrainingParams']['nb_steps']))
    
    for i in range(np.shape(init_params['TrainingParams']['nb_steps'])[0]):

        b_inf = int(np.sum(init_params['TrainingParams']['nb_steps'][:i]))
        b_sup = np.sum(init_params['TrainingParams']['nb_steps'][:i+1])

        learning_rate_list[b_inf:b_sup] = init_params['TrainingParams']['learning_rates'][i]

    params['TrainingParams']['learning_rate_list'] = learning_rate_list
    
    if params['NetworkParams']['Loss'] in ['cross_entropy_softDice', 'weighted_cross_entropy_softDice']: 
        weight_dice_list[params['NetworkParams']['softDice_start']:] = 1
    
    params['TrainingParams']['weight_dice_list'] = weight_dice_list
        
# Save codes and initialize the environment 

    params['TrainingParams']['path_root'] = 'savenet_/' # Main directory to save training/validation results
    pythonfnames = ['augmentation.py',
                    'convenience.py',
                    'cost_functions.py',
                    'dataset.py',
                    'gaussian_random_fields.py',
                    'global_initialization.py',
                    'main_train_Unet.py',
                    'main_train_Vnet.py',
                    'main_train_Unet.ipynb',
                    'main_train_Vnet.ipynb',
                    'net3d.py',
                    'network.py',
                    'neuralnets.py',
                    'parameters_initialization.py',
                    'test.py',
                    'train.py'] # Python files to save
        
    # Create a new directory

    params['TrainingParams']['path_save'] = params['DataParams']['filepathsave'] + params['TrainingParams']['path_root'] + time.strftime('%b_%d(%Y)-%H_%M_%S') + "/"
    params['TrainingParams']['learning_path'] = params['TrainingParams']['path_save'] + "learning/"
    params['TrainingParams']['results_path'] = params['TrainingParams']['path_save'] + "results/"
    params['TrainingParams']['pythonfiles_path'] = params['TrainingParams']['path_save'] + "pythonfiles/"
    
    if os.path.isdir(params['TrainingParams']['path_root']) == False:
        os.mkdir(params['TrainingParams']['path_root'])

    os.mkdir(params['TrainingParams']['path_save'])
    os.mkdir(params['TrainingParams']['learning_path'])
    os.mkdir(params['TrainingParams']['results_path'])
    os.mkdir(params['TrainingParams']['pythonfiles_path'])
        
    # Creation of the dataset     
    
    data = dataset.FullDataset(images, labels, params['DataParams'])
        
    with open(params['TrainingParams']['path_save'] + 'results/' + 'params.pickle', 'wb') as file:
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(params)

    for pythonfname in pythonfnames:
        shutil.copyfile(pythonfname, params['TrainingParams']['pythonfiles_path'] + pythonfname)
        
    return data, params
  

=======
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import os
import shutil
import time
import pickle
from src.data import dataset
import scipy.io as sio
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_database_matfile(params):
    """ Create dataset from several matlab file """
    
    filepathname = params['DataParams']['filepathname']
    data_list = params['DataParams']['data_list']
    
    images = []
    labels = []

    print('Loaded data :\n')
    
    for i in range(len(data_list)):
        
        data = sio.loadmat(filepathname + data_list[i] + '.mat')  
        data['volume'] = data['volume'].astype(np.uint8)   
        data['label'] = data['label'].astype(np.uint8)
        
        data_temp = data['volume']
           
        if params['NetworkParams']['Network'] in  ['UNet_CPPN']:
            
            data_norm = np.zeros([data_temp.shape[0], data_temp.shape[1], data_temp.shape[2]])
            for j in range(data_norm.shape[0]):
                if data_temp[j,:,:].std() != 0:
                    data_norm[j,:,:] = (data_temp[j,:,:] - data_temp[j,:,:].mean())/data_temp[j,:,:].std()

        elif params['NetworkParams']['Network'] in ['VNet_CPPN']:
        
            data_norm = (data_temp - data_temp.mean())/data_temp.std()
        
        images.append(data_norm)
        labels.append(data['label'])
        
        if (os.path.isfile(filepathname + data_list[i] + '_image.nii') or os.path.isfile(filepathname + data_list[i] + '_label.nii')) == False:
            
            image_loc = nib.Nifti1Image(np.int16(np.transpose(data['volume'], (2,0,1))), -np.eye(4))
            #nib.save(image_loc, filepathname + data_list[i] + '_image.nii')
            
            for j in range(max(data['label'].reshape(-1))):
                
                label_loc = data['label'] == j + 1
                label_loc = nib.Nifti1Image(np.int16(np.transpose(label_loc, (2,0,1))), -np.eye(4))
                
                #nib.save(label_loc, filepathname + data_list[i] + '_label_' + str(j + 1) + '.nii')
            
            #del image_loc, label_loc

        del data, data_temp, data_norm 
        
        print(filepathname + data_list[i])
        
    return images, labels



def read_data_set(init_params):
    """ The exposed read function. Fetch the directory in the file dirfile.txt
        tagstr : a string containing the tag of the dataset to load (default, '1')
    """
    
    params = copy.copy(init_params)
    
# Load dataset

    #lowres_data, highres_data, label = load_one_matfile(params['DataParams']['filepathname'])
    images, labels = load_database_matfile(params)

    
# Creation of a dictionnary which contains all the network parameters 
    
    # Set parameters from loaded data
    params['DataParams']['num_volumes'] = len(images)    
    params['DataParams']['patchxin'] = params['DataParams']['patch_size'][1] + params['DataParams']['augx'] * 2
    params['DataParams']['patchyin'] = params['DataParams']['patch_size'][2] + params['DataParams']['augx'] * 2
    params['DataParams']['patchdepth'] = params['DataParams']['patch_size'][0]
    params['DataParams']['patchxout'] = params['DataParams']['patch_size'][1] 
    params['DataParams']['patchyout'] = params['DataParams']['patch_size'][2]
    params['DataParams']['num_classes'] = np.asarray(labels).reshape(-1).max() + 1 
    
    # Definition of the gradient step and Dice loss start
    learning_rate_list = np.zeros(np.sum(params['TrainingParams']['nb_steps']))
    weight_dice_list = np.zeros(np.sum(params['TrainingParams']['nb_steps']))
    
    for i in range(np.shape(init_params['TrainingParams']['nb_steps'])[0]):

        b_inf = int(np.sum(init_params['TrainingParams']['nb_steps'][:i]))
        b_sup = np.sum(init_params['TrainingParams']['nb_steps'][:i+1])

        learning_rate_list[b_inf:b_sup] = init_params['TrainingParams']['learning_rates'][i]

    params['TrainingParams']['learning_rate_list'] = learning_rate_list
    
    if params['NetworkParams']['Loss'] in ['cross_entropy_softDice', 'weighted_cross_entropy_softDice']: 
        weight_dice_list[params['NetworkParams']['softDice_start']:] = 1
    
    params['TrainingParams']['weight_dice_list'] = weight_dice_list
        
# Save codes and initialize the environment 

    params['TrainingParams']['path_root'] = 'savenet_/' # Main directory to save training/validation results
    pythonfnames = ['augmentation.py',
                    'convenience.py',
                    'cost_functions.py',
                    'dataset.py',
                    'gaussian_random_fields.py',
                    'global_initialization.py',
                    'main_train_Unet.py',
                    'main_train_Vnet.py',
                    'main_train_Unet.ipynb',
                    'main_train_Vnet.ipynb',
                    'net3d.py',
                    'network.py',
                    'neuralnets.py',
                    'parameters_initialization.py',
                    'test.py',
                    'train.py'] # Python files to save
        
    # Create a new directory

    params['TrainingParams']['path_save'] = params['DataParams']['filepathsave'] + params['TrainingParams']['path_root'] + time.strftime('%b_%d(%Y)-%H_%M_%S') + "/"
    params['TrainingParams']['learning_path'] = params['TrainingParams']['path_save'] + "learning/"
    params['TrainingParams']['results_path'] = params['TrainingParams']['path_save'] + "results/"
    params['TrainingParams']['pythonfiles_path'] = params['TrainingParams']['path_save'] + "pythonfiles/"
    
    if os.path.isdir(params['TrainingParams']['path_root']) == False:
        os.mkdir(params['TrainingParams']['path_root'])

    os.mkdir(params['TrainingParams']['path_save'])
    os.mkdir(params['TrainingParams']['learning_path'])
    os.mkdir(params['TrainingParams']['results_path'])
    os.mkdir(params['TrainingParams']['pythonfiles_path'])
        
    # Creation of the dataset     
    
    data = dataset.FullDataset(images, labels, params['DataParams'])
        
    with open(params['TrainingParams']['path_save'] + 'results/' + 'params.pickle', 'wb') as file:
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(params)

    for pythonfname in pythonfnames:
        shutil.copyfile(pythonfname, params['TrainingParams']['pythonfiles_path'] + pythonfname)
        
    return data, params
  

>>>>>>> theirs
