<<<<<<< ours
import os
import numpy as np
import scipy.io as sio
from tools.export_rawmhd import export_rawmhd
from tools.MITKviewer import valider_recalage
#import nrrd
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
import logging
from tools.export_to_3d_dicom import export_to_3d_dicom


# 1. Configuration de base (format et niveau de priorité)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Création de l'instance du logger pour ce fichier
logger = logging.getLogger(__name__)

def save_registered_data(d, data_recal, main_dir, num_patient, ref, t_form):

    app = QApplication.instance()
    
    output_dir = os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Recalage', ref)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Enregistrement des données pour {ref}...")
    
    # --- 1. PRÉPARATION DES MATRICES (MATLAB STRUCT) ---
    num_images = data_recal.shape[2]
    tform_list = []

    for i in range(num_images):
        if i < len(t_form):
            tf_matrix = t_form[i]
            tx, ty = tf_matrix[0, 2], tf_matrix[1, 2]
            angle = np.degrees(np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]))

            entry = {
                'T': tf_matrix.T,
                'R': tf_matrix[0:2, 0:2],
                'Translation': [tx, ty],
                'RotationAngle': angle,
                'Dimensionality': 2,
                'A': tf_matrix,
                'Correlation': 1.0
            }
        else:
            entry = {
                'T': np.eye(3), 'R': np.eye(2), 'Translation': [0.0, 0.0],
                'RotationAngle': 0.0, 'Dimensionality': 2, 'A': np.eye(3), 'Correlation': 0.0
            }
        tform_list.append(entry)
        
        # Optimisation GUI : rafraîchir toutes les 50 images pour ne pas ramer
        if i % 50 == 0 and app:
            app.processEvents()

    tform_struct_array = np.array(tform_list, dtype=object)

    # --- 2. SAUVEGARDE MATLAB ---
    mat_path = os.path.join(output_dir, f"data_recal_{ref}_d_{d}.mat")
    sio.savemat(mat_path, {
        'data_recal': data_recal,
        't_form': tform_struct_array
    }, appendmat=False, do_compression=True)

    # --- 3. EXPORT MHD/RAW (SÉCURISÉ) ---
    # Sécurité max pour éviter la division par zéro
    max_val = np.max(data_recal)
    if max_val == 0:
        max_val = 1.0
        logger.warning(f"Attention: Volume vide pour {ref}")

    # Conversion propre en uint8 pour MITK/MITK
    data_uint8 = (data_recal.astype(np.float32) / max_val * 255).astype(np.uint8)
    
    # 3. Export vers le DICOM 3D Unique (Utilise les données 0-255)
    # On passe les arguments explicitement pour éviter les erreurs d'inversion
    output_dicom_file = os.path.join(output_dir, f"data_recal_{ref}_d_{d}.dcm")

    export_to_3d_dicom(
        data=data_uint8, 
        output_filepath=output_dicom_file, 
        patient_name=str(num_patient), 
        patient_id=str(ref)
    )
    
    valider_recalage(data_recal)
                  
    if app: app.processEvents()
    print(f"Sauvegarde terminée avec succès dans : {output_dir}")


=======
import os
import numpy as np
import scipy.io as sio
from tools.export_rawmhd import export_rawmhd
from tools.MITKviewer import valider_recalage
#import nrrd
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
import logging
from tools.export_to_3d_dicom import export_to_3d_dicom


# 1. Configuration de base (format et niveau de priorité)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Création de l'instance du logger pour ce fichier
logger = logging.getLogger(__name__)

def save_registered_data(d, data_recal, main_dir, num_patient, ref, t_form):

    app = QApplication.instance()
    
    output_dir = os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Recalage', ref)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Enregistrement des données pour {ref}...")
    
    # --- 1. PRÉPARATION DES MATRICES (MATLAB STRUCT) ---
    num_images = data_recal.shape[2]
    tform_list = []

    for i in range(num_images):
        if i < len(t_form):
            tf_matrix = t_form[i]
            tx, ty = tf_matrix[0, 2], tf_matrix[1, 2]
            angle = np.degrees(np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]))

            entry = {
                'T': tf_matrix.T,
                'R': tf_matrix[0:2, 0:2],
                'Translation': [tx, ty],
                'RotationAngle': angle,
                'Dimensionality': 2,
                'A': tf_matrix,
                'Correlation': 1.0
            }
        else:
            entry = {
                'T': np.eye(3), 'R': np.eye(2), 'Translation': [0.0, 0.0],
                'RotationAngle': 0.0, 'Dimensionality': 2, 'A': np.eye(3), 'Correlation': 0.0
            }
        tform_list.append(entry)
        
        # Optimisation GUI : rafraîchir toutes les 50 images pour ne pas ramer
        if i % 50 == 0 and app:
            app.processEvents()

    tform_struct_array = np.array(tform_list, dtype=object)

    # --- 2. SAUVEGARDE MATLAB ---
    mat_path = os.path.join(output_dir, f"data_recal_{ref}_d_{d}.mat")
    sio.savemat(mat_path, {
        'data_recal': data_recal,
        't_form': tform_struct_array
    }, appendmat=False, do_compression=True)

    # --- 3. EXPORT MHD/RAW (SÉCURISÉ) ---
    # Sécurité max pour éviter la division par zéro
    max_val = np.max(data_recal)
    if max_val == 0:
        max_val = 1.0
        logger.warning(f"Attention: Volume vide pour {ref}")

    # Conversion propre en uint8 pour MITK/MITK
    data_uint8 = (data_recal.astype(np.float32) / max_val * 255).astype(np.uint8)
    
    # 3. Export vers le DICOM 3D Unique (Utilise les données 0-255)
    # On passe les arguments explicitement pour éviter les erreurs d'inversion
    output_dicom_file = os.path.join(output_dir, f"data_recal_{ref}_d_{d}.dcm")

    export_to_3d_dicom(
        data=data_uint8, 
        output_filepath=output_dicom_file, 
        patient_name=str(num_patient), 
        patient_id=str(ref)
    )
    
    valider_recalage(data_recal)
                  
    if app: app.processEvents()
    print(f"Sauvegarde terminée avec succès dans : {output_dir}")


>>>>>>> theirs
