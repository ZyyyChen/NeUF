<<<<<<< ours
import os
import pickle
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path

from lib.segAuto.src.training import global_initialization
from lib.segAuto.src.inference import _test
from lib.segAuto.src.data import dataset
from lib.segAuto.src.inference import Vnet_inference
from tools.MITKviewer import valider_recalage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optionnel : désactive explicitement oneDNN


def run_inference(volume_path, data_name, affine, preprocess_info, session, maindir, ref):
    save_path = str(Path(maindir) / "SegAuto_VNet")
    os.makedirs(save_path, exist_ok=True)
    
    training_results_path = os.path.join(maindir, 'savenet_')
  
    network_path = os.path.join(training_results_path, session)

    with open(os.path.join(network_path, 'results', 'params.pickle'), 'rb') as file:
        params = pickle.load(file)

    params['DataParams']['data_list'] = data_name
    params['DataParams']['filepathname'] = os.path.dirname(volume_path) + '/'
    params['DataParams']['filepathsave'] = os.getcwd() + '/'
    params['TrainingParams']['path_save'] = network_path
    params['TrainingParams']['results_path'] = os.path.join(network_path, 'results/')

    print('volume_path',str(volume_path))
    images, labels = Vnet_inference.LoadVolume(maindir, volume_path)
    data = dataset.FullDataset(images, labels, params['DataParams'])

    global_initialization.initialize_network(params)
    savetrain, saveval = global_initialization.load_best_network(params)

    # Inject placeholders and session into the _test module for seg_volume
    for var in ['x_', 'keep_prob_', 'is_train', 'y_', 'cl_lay', 'y_soft', 'frequencies_', 'weight_dice', 'learning_rate', 'sess']:
        if hasattr(global_initialization, var):
            setattr(_test, var, getattr(global_initialization, var))

    test_params = {
        'test_patch_depth': 64,
        'recouv': 75,
        'data2seg': data_name,
        'session': session,
        'save_path': save_path,
        'affine': affine,
        'preprocess_info': preprocess_info,
    }

    seg, results = _test.seg_volume(params, test_params, data)
    print("OK.....")

    # Preview volume reconst 3D + contour du mask 3D
    vol_rescont =  pydicom.dcmread(os.path.join(maindir, 'Pre_traitement_echo_v2', 'Repere_commun', ref, f'data_repcom_{ref}.dcm'))
    vol_rescont = vol_rescont.pixel_array
    vol_rescont = np.transpose(vol_rescont, (0, 2, 1))
    vol_rescont = vol_rescont[::-1,::-1,::-1]
    vol_rescont = vol_rescont[:,::-1,::-1]
    
    seg_mask = nib.load(os.path.join(maindir, 'SegAuto_VNet', ref,f'{ref}_auto_seg_contour_1.nii.gz')).get_fdata()
    seg_mask = np.transpose(seg_mask, (2, 0, 1))
    seg_mask = seg_mask[::-1,::-1,::-1]
    seg_mask = seg_mask[:,::-1,::-1]
    
    valider_recalage(vol_rescont, seg_mask)
    
    return seg, results
=======
import os
import pickle
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path

from lib.segAuto.src.training import global_initialization
from lib.segAuto.src.inference import _test
from lib.segAuto.src.data import dataset
from lib.segAuto.src.inference import Vnet_inference
from tools.MITKviewer import valider_recalage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optionnel : désactive explicitement oneDNN


def run_inference(volume_path, data_name, affine, preprocess_info, session, maindir, ref):
    save_path = str(Path(maindir) / "SegAuto_VNet")
    os.makedirs(save_path, exist_ok=True)
    
    training_results_path = os.path.join(maindir, 'savenet_')
  
    network_path = os.path.join(training_results_path, session)

    with open(os.path.join(network_path, 'results', 'params.pickle'), 'rb') as file:
        params = pickle.load(file)

    params['DataParams']['data_list'] = data_name
    params['DataParams']['filepathname'] = os.path.dirname(volume_path) + '/'
    params['DataParams']['filepathsave'] = os.getcwd() + '/'
    params['TrainingParams']['path_save'] = network_path
    params['TrainingParams']['results_path'] = os.path.join(network_path, 'results/')

    print('volume_path',str(volume_path))
    images, labels = Vnet_inference.LoadVolume(maindir, volume_path)
    data = dataset.FullDataset(images, labels, params['DataParams'])

    global_initialization.initialize_network(params)
    savetrain, saveval = global_initialization.load_best_network(params)

    # Inject placeholders and session into the _test module for seg_volume
    for var in ['x_', 'keep_prob_', 'is_train', 'y_', 'cl_lay', 'y_soft', 'frequencies_', 'weight_dice', 'learning_rate', 'sess']:
        if hasattr(global_initialization, var):
            setattr(_test, var, getattr(global_initialization, var))

    test_params = {
        'test_patch_depth': 64,
        'recouv': 75,
        'data2seg': data_name,
        'session': session,
        'save_path': save_path,
        'affine': affine,
        'preprocess_info': preprocess_info,
    }

    seg, results = _test.seg_volume(params, test_params, data)
    print("OK.....")

    # Preview volume reconst 3D + contour du mask 3D
    vol_rescont =  pydicom.dcmread(os.path.join(maindir, 'Pre_traitement_echo_v2', 'Repere_commun', ref, f'data_repcom_{ref}.dcm'))
    vol_rescont = vol_rescont.pixel_array
    vol_rescont = np.transpose(vol_rescont, (0, 2, 1))
    vol_rescont = vol_rescont[::-1,::-1,::-1]
    vol_rescont = vol_rescont[:,::-1,::-1]
    
    seg_mask = nib.load(os.path.join(maindir, 'SegAuto_VNet', ref,f'{ref}_auto_seg_contour_1.nii.gz')).get_fdata()
    seg_mask = np.transpose(seg_mask, (2, 0, 1))
    seg_mask = seg_mask[::-1,::-1,::-1]
    seg_mask = seg_mask[:,::-1,::-1]
    
    valider_recalage(vol_rescont, seg_mask)
    
    return seg, results
>>>>>>> theirs
