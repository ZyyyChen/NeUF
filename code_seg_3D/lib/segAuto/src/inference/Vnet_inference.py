import os
import numpy as np
import nibabel as ni
from pathlib import Path
from skimage.transform import resize
import pydicom
from lib.segAuto.src.inference import inferenceV3
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optionnel : désactive explicitement oneDNN

volume_path = None
data_name = None
affine = None
preprocess_info = None

session = 'SavedModel_après_correction_contour'

def SetVolumePath(vol_path, _data_name):
    global volume_path, data_name, affine, preprocess_info
    volume_path = vol_path
    data_name = _data_name
    affine = None
    preprocess_info = None


def LoadVolume(maindir, volume_path):
    global affine, preprocess_info
    volume_path = str(volume_path)
    if volume_path.endswith('.dcm'):
        # Read the DICOM metadata and pixel data
        ds = pydicom.dcmread(volume_path)
        
        # Access the pixel data as a NumPy array
        vol = ds.pixel_array
        vol = np.transpose(vol, (2, 1, 0))
        
        """valider_recalage(vol)
        vol = np.transpose(vol, (0, 2, 1))
        vol = vol[::-1,::-1,::-1]
        vol = vol[:,::-1,::-1]"""
    else:
        raise TypeError('input file must be .dcm')

    
    prefix = os.path.splitext(data_name)[0].replace('data_repcom_', '')
    output_dir = Path(maindir) / "newold_dataset_640_resize_320" / prefix
    

    vol = vol.astype(np.uint8)
    original_shape = tuple(int(v) for v in vol.shape)
    size = 320
    pad_width = ((300, 300), (300, 300), (300, 300))
    vol = np.pad(vol, pad_width)
    center = (np.array(vol.shape) / 2).astype(np.int32)
    vol = vol[center[0]-size:center[0]+size, center[1]-size:center[1]+size, center[2]-size:center[2]+size]
    vol = resize(vol, (size, size, size), anti_aliasing=False, order=0)

    preprocess_info = {
        'original_shape': original_shape,
        'pad_width': pad_width,
        'crop_start': tuple(int(c - size) for c in center),
        'crop_end': tuple(int(c + size) for c in center),
        'crop_shape': (2*size,2*size,2*size),
        'resize_shape': (size,size,size),
    }

    vol = (vol - vol.mean()) / vol.std()
    s = vol.shape
    label = np.zeros(s)
    label[int(s[0]/2)-10:int(s[0]/2)+10, int(s[1]/2)-10:int(s[1]/2)+10, int(s[2]/2)-10:int(s[2]/2)+10] = 1
    label = (label > 0).astype(np.uint8)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
  
    ni.save(ni.Nifti1Image(vol.astype(np.float32), affine), str(out / f"{prefix}_vol_320.nii.gz"))
    ni.save(ni.Nifti1Image(label.astype(np.uint8), affine), str(out / f"{prefix}_label_dummy_320.nii.gz"))
    
    return [vol], [label]


def Inference(maindir, ref):
    inferenceV3.run_inference(volume_path, data_name, affine, preprocess_info, session, maindir, ref)
