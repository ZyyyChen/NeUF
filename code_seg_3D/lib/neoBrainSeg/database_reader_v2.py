<<<<<<< ours
import pydicom
import numpy as np
import os
from pathlib import Path

def database_reader_v2(acq_dir, main_dir, num_patient, ref_seq_dyn, ref_img_sag):
    """
    Reads DICOM images for sagittal reference and dynamic sequences.
    Adjusts orientation based on acquisition direction.
    """
    
    # Normalize and validate main_dir to avoid malformed paths on Windows
    if not isinstance(main_dir, str):
        main_dir = str(main_dir)
   
    # 1. Nettoyage et normalisation du chemin racine
    # resolve() transforme "\Users" en "C:\Users" et nettoie les colons parasites
    base_dir = Path(main_dir.lstrip(':')).resolve()

    # 2. Construction du chemin dossier (syntaxe / très propre avec pathlib)
    fdir = base_dir / 'Examen_echographie' / str(num_patient)

    # 3. Validation du nom de fichier
    filename_sag = str(ref_img_sag).strip()
   
    if not filename_sag or filename_sag.lower() == "nan":
        raise FileNotFoundError(f"Nom de fichier sagittal manquant pour le patient {num_patient}.")

    # 4. Vérification d'existence et lecture
    # Remplace ou ajoute l'extension .dcm
    path_sag = (fdir / filename_sag).with_suffix('.dcm')

    if not path_sag.exists():
        # .absolute() permet d'afficher le chemin complet pour faciliter le debug
        raise FileNotFoundError(f"DICOM Sagittal introuvable : {path_sag.absolute()}")

    # Lecture du DICOM
    ds_sag = pydicom.dcmread(str(path_sag))
    data_sag = ds_sag.pixel_array
    
    # Note: pydicom may return (Rows, Cols, Channels) for RGB data
    # We need to handle the color channels
    
    # MATLAB: squeeze(data_sag(:,:,1,:))
    # In Python/NumPy, DICOM pixel_array is usually (Rows, Cols) or (Rows, Cols, Channels)
    if data_sag.ndim == 3:
        # RGB case: (Rows, Cols, Channels) - extract first channel to match MATLAB behavior
        data_sag = data_sag[:, :, 0]  # Result is (Rows, Cols)
    elif data_sag.ndim > 3:
        # Multi-frame case: squeeze and then extract channel if needed
        data_sag = np.squeeze(data_sag)
        if data_sag.ndim == 3:
            data_sag = data_sag[:, :, 0]  # Result is (Rows, Cols)
    

    # 2. Read Dynamic Sequence
    # Ensure the dynamic sequence filename is present
    if not ref_seq_dyn or str(ref_seq_dyn).strip() == "":
        raise FileNotFoundError(f"Sequence filename is missing for patient {num_patient}. Check reference sheet.")
    # 1. Utiliser pathlib pour construire le chemin
    # fdir doit déjà être un objet Path ou une chaîne propre
    filename = f"{str(ref_seq_dyn).strip()}.dcm"
    path_seq = Path(fdir) / filename

    # 2. Vérification d'existence
    if not path_seq.exists():
        raise FileNotFoundError(f"Sequence DICOM not found: {path_seq.absolute()}")

    # 3. Lecture (pydicom accepte un objet Path, mais str() est parfois plus sûr)
    ds_seq = pydicom.dcmread(str(path_seq))
    data_seqdyn = ds_seq.pixel_array
    
    # Note: pydicom returns (Frames, Rows, Cols, [Channels])
    # MATLAB displays as (Rows, Cols, Frames), so axes are permuted
    
    if data_seqdyn.ndim > 2:
        data_seqdyn = np.squeeze(data_seqdyn)
        # Convert from pydicom's (Frames, Rows, Cols) to MATLAB's (Rows, Cols, Frames)
        # This handles both RGB (4D) and grayscale (3D) data
        if data_seqdyn.ndim == 4:
            # RGB case: (Frames, Rows, Cols, Channels) -> extract first channel -> (Rows, Cols, Frames)
            data_seqdyn = np.transpose(data_seqdyn, (1, 2, 0, 3))  # First transpose to (Rows, Cols, Frames, Channels)
            data_seqdyn = data_seqdyn[:, :, :, 0]  # Extract first channel -> (Rows, Cols, Frames)
        elif data_seqdyn.ndim == 3:
            # Grayscale case: (Frames, Rows, Cols) -> (Rows, Cols, Frames)
            data_seqdyn = np.transpose(data_seqdyn, (1, 2, 0))
    

    # 3. Correction of orientation (acq_dir == 'P-A')
    # MATLAB flip(data, 3) flips the 3rd dimension (frames/slices)
    # After transpose, data_seqdyn is now (Rows, Cols, Frames, [Channels])
    # so flipping axis 2 (Frames) matches MATLAB's flip(data, 3)
    
    if acq_dir == 'P-A':
        # Inversion de l'axe 2 de manière ultra-rapide (O(1) complexity)
        data_seqdyn = data_seqdyn[:, :, ::-1, ...]
        
        # Facultatif : Force la création d'un tableau contigu si vous faites 
        # des calculs lourds juste après (comme le recalage).
        data_seqdyn = np.ascontiguousarray(data_seqdyn)

    return data_sag, data_seqdyn
=======
import pydicom
import numpy as np
import os
from pathlib import Path

def database_reader_v2(acq_dir, main_dir, num_patient, ref_seq_dyn, ref_img_sag):
    """
    Reads DICOM images for sagittal reference and dynamic sequences.
    Adjusts orientation based on acquisition direction.
    """
    
    # Normalize and validate main_dir to avoid malformed paths on Windows
    if not isinstance(main_dir, str):
        main_dir = str(main_dir)
   
    # 1. Nettoyage et normalisation du chemin racine
    # resolve() transforme "\Users" en "C:\Users" et nettoie les colons parasites
    base_dir = Path(main_dir.lstrip(':')).resolve()

    # 2. Construction du chemin dossier (syntaxe / très propre avec pathlib)
    fdir = base_dir / 'Examen_echographie' / str(num_patient)

    # 3. Validation du nom de fichier
    filename_sag = str(ref_img_sag).strip()
   
    if not filename_sag or filename_sag.lower() == "nan":
        raise FileNotFoundError(f"Nom de fichier sagittal manquant pour le patient {num_patient}.")

    # 4. Vérification d'existence et lecture
    # Remplace ou ajoute l'extension .dcm
    path_sag = (fdir / filename_sag).with_suffix('.dcm')

    if not path_sag.exists():
        # .absolute() permet d'afficher le chemin complet pour faciliter le debug
        raise FileNotFoundError(f"DICOM Sagittal introuvable : {path_sag.absolute()}")

    # Lecture du DICOM
    ds_sag = pydicom.dcmread(str(path_sag))
    data_sag = ds_sag.pixel_array
    
    # Note: pydicom may return (Rows, Cols, Channels) for RGB data
    # We need to handle the color channels
    
    # MATLAB: squeeze(data_sag(:,:,1,:))
    # In Python/NumPy, DICOM pixel_array is usually (Rows, Cols) or (Rows, Cols, Channels)
    if data_sag.ndim == 3:
        # RGB case: (Rows, Cols, Channels) - extract first channel to match MATLAB behavior
        data_sag = data_sag[:, :, 0]  # Result is (Rows, Cols)
    elif data_sag.ndim > 3:
        # Multi-frame case: squeeze and then extract channel if needed
        data_sag = np.squeeze(data_sag)
        if data_sag.ndim == 3:
            data_sag = data_sag[:, :, 0]  # Result is (Rows, Cols)
    

    # 2. Read Dynamic Sequence
    # Ensure the dynamic sequence filename is present
    if not ref_seq_dyn or str(ref_seq_dyn).strip() == "":
        raise FileNotFoundError(f"Sequence filename is missing for patient {num_patient}. Check reference sheet.")
    # 1. Utiliser pathlib pour construire le chemin
    # fdir doit déjà être un objet Path ou une chaîne propre
    filename = f"{str(ref_seq_dyn).strip()}.dcm"
    path_seq = Path(fdir) / filename

    # 2. Vérification d'existence
    if not path_seq.exists():
        raise FileNotFoundError(f"Sequence DICOM not found: {path_seq.absolute()}")

    # 3. Lecture (pydicom accepte un objet Path, mais str() est parfois plus sûr)
    ds_seq = pydicom.dcmread(str(path_seq))
    data_seqdyn = ds_seq.pixel_array
    
    # Note: pydicom returns (Frames, Rows, Cols, [Channels])
    # MATLAB displays as (Rows, Cols, Frames), so axes are permuted
    
    if data_seqdyn.ndim > 2:
        data_seqdyn = np.squeeze(data_seqdyn)
        # Convert from pydicom's (Frames, Rows, Cols) to MATLAB's (Rows, Cols, Frames)
        # This handles both RGB (4D) and grayscale (3D) data
        if data_seqdyn.ndim == 4:
            # RGB case: (Frames, Rows, Cols, Channels) -> extract first channel -> (Rows, Cols, Frames)
            data_seqdyn = np.transpose(data_seqdyn, (1, 2, 0, 3))  # First transpose to (Rows, Cols, Frames, Channels)
            data_seqdyn = data_seqdyn[:, :, :, 0]  # Extract first channel -> (Rows, Cols, Frames)
        elif data_seqdyn.ndim == 3:
            # Grayscale case: (Frames, Rows, Cols) -> (Rows, Cols, Frames)
            data_seqdyn = np.transpose(data_seqdyn, (1, 2, 0))
    

    # 3. Correction of orientation (acq_dir == 'P-A')
    # MATLAB flip(data, 3) flips the 3rd dimension (frames/slices)
    # After transpose, data_seqdyn is now (Rows, Cols, Frames, [Channels])
    # so flipping axis 2 (Frames) matches MATLAB's flip(data, 3)
    
    if acq_dir == 'P-A':
        # Inversion de l'axe 2 de manière ultra-rapide (O(1) complexity)
        data_seqdyn = data_seqdyn[:, :, ::-1, ...]
        
        # Facultatif : Force la création d'un tableau contigu si vous faites 
        # des calculs lourds juste après (comme le recalage).
        data_seqdyn = np.ascontiguousarray(data_seqdyn)

    return data_sag, data_seqdyn
>>>>>>> theirs
