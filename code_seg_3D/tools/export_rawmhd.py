<<<<<<< ours
import os
import numpy as np
import matplotlib.pyplot as plt

def export_rawmhd(data, filepath, filename, raw_storage='uint8', flag_postprocessing=1):

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # 1. Post-processing (Normalisation 0-1)
    if flag_postprocessing:
        data = data.astype(np.float32)
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        else:
            data = data - d_min

    # 2. Type de données et StringType MHD (Correction pour uint8)
    if raw_storage in ['int8', 'uint8']:
        # Si la donnée est déjà normalisée (0-1), on scale à 255
        if np.max(data) <= 1.01:
            data_to_write = (data * 255).round().astype(np.uint8)
        else:
            data_to_write = data.astype(np.uint8)
        
        # MET_UCHAR est indispensable pour une luminosité correcte (0-255 non signé)
        stringtype = 'MET_UCHAR'
        
    elif raw_storage == 'double':
        data_to_write = data.astype(np.float64)
        stringtype = 'MET_DOUBLE'
    else:
        raise ValueError(f"Type '{raw_storage}' non supporté. Utilisez 'uint8', 'int8' ou 'double'.")

    # 3. Gestion du nombre de canaux et orientation
    if data_to_write.ndim == 3:
        nch = 1
        # Transposition pour l'ordre de lecture MITK (Z, Y, X)
        data_to_write = data_to_write.transpose(0, 2, 1)
        data_to_write = np.flip(data_to_write, axis=0)
    elif data_to_write.ndim == 4:
        nch = data_to_write.shape[3]
        data_to_write = data_to_write.transpose(0, 2, 1, 3)
        data_to_write = np.flip(data_to_write, axis=0)
    else:
        raise ValueError("La donnée doit être 3D ou 4D.")
    

    # 4. Préparation Mémoire
    data_to_write = np.ascontiguousarray(data_to_write)
    shape_py = data_to_write.shape
    
    # 5. Écriture du fichier RAW
    rawfile_name = filename + '.raw'
    rawfile_path = os.path.join(filepath, rawfile_name)
    data_to_write.tofile(rawfile_path)

    # 6. Écriture du fichier MHD
    mhdfile_path = os.path.join(filepath, filename + '.mhd')
    with open(mhdfile_path, 'w') as f:
        f.write('ObjectType = Image\n')
        f.write('NDims = 3\n')
        f.write('BinaryData = True\n')
        f.write('BinaryDataByteOrderMSB = False\n')
        f.write('CompressedData = False\n')
        f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
        f.write('AnatomicalOrientation = RAI\n')
        f.write('ElementSpacing = 1 1 1\n')
        f.write('Offset = 0 0 0\n')
        # DimSize : X Y Z
        f.write(f'DimSize = {shape_py[2]} {shape_py[1]} {shape_py[0]}\n')
        f.write(f'ElementNumberOfChannels = {nch}\n')
        f.write(f'ElementType = {stringtype}\n')
        f.write(f'ElementDataFile = {rawfile_name}\n')
        f.write('ElementMin = 0\n')
        f.write('ElementMax = 255\n')

    return mhdfile_path

=======
import os
import numpy as np
import matplotlib.pyplot as plt

def export_rawmhd(data, filepath, filename, raw_storage='uint8', flag_postprocessing=1):

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # 1. Post-processing (Normalisation 0-1)
    if flag_postprocessing:
        data = data.astype(np.float32)
        d_min = np.min(data)
        d_max = np.max(data)
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        else:
            data = data - d_min

    # 2. Type de données et StringType MHD (Correction pour uint8)
    if raw_storage in ['int8', 'uint8']:
        # Si la donnée est déjà normalisée (0-1), on scale à 255
        if np.max(data) <= 1.01:
            data_to_write = (data * 255).round().astype(np.uint8)
        else:
            data_to_write = data.astype(np.uint8)
        
        # MET_UCHAR est indispensable pour une luminosité correcte (0-255 non signé)
        stringtype = 'MET_UCHAR'
        
    elif raw_storage == 'double':
        data_to_write = data.astype(np.float64)
        stringtype = 'MET_DOUBLE'
    else:
        raise ValueError(f"Type '{raw_storage}' non supporté. Utilisez 'uint8', 'int8' ou 'double'.")

    # 3. Gestion du nombre de canaux et orientation
    if data_to_write.ndim == 3:
        nch = 1
        # Transposition pour l'ordre de lecture MITK (Z, Y, X)
        data_to_write = data_to_write.transpose(0, 2, 1)
        data_to_write = np.flip(data_to_write, axis=0)
    elif data_to_write.ndim == 4:
        nch = data_to_write.shape[3]
        data_to_write = data_to_write.transpose(0, 2, 1, 3)
        data_to_write = np.flip(data_to_write, axis=0)
    else:
        raise ValueError("La donnée doit être 3D ou 4D.")
    

    # 4. Préparation Mémoire
    data_to_write = np.ascontiguousarray(data_to_write)
    shape_py = data_to_write.shape
    
    # 5. Écriture du fichier RAW
    rawfile_name = filename + '.raw'
    rawfile_path = os.path.join(filepath, rawfile_name)
    data_to_write.tofile(rawfile_path)

    # 6. Écriture du fichier MHD
    mhdfile_path = os.path.join(filepath, filename + '.mhd')
    with open(mhdfile_path, 'w') as f:
        f.write('ObjectType = Image\n')
        f.write('NDims = 3\n')
        f.write('BinaryData = True\n')
        f.write('BinaryDataByteOrderMSB = False\n')
        f.write('CompressedData = False\n')
        f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
        f.write('AnatomicalOrientation = RAI\n')
        f.write('ElementSpacing = 1 1 1\n')
        f.write('Offset = 0 0 0\n')
        # DimSize : X Y Z
        f.write(f'DimSize = {shape_py[2]} {shape_py[1]} {shape_py[0]}\n')
        f.write(f'ElementNumberOfChannels = {nch}\n')
        f.write(f'ElementType = {stringtype}\n')
        f.write(f'ElementDataFile = {rawfile_name}\n')
        f.write('ElementMin = 0\n')
        f.write('ElementMax = 255\n')

    return mhdfile_path

>>>>>>> theirs
