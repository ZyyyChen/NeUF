<<<<<<< ours
import os
import scipy.io as sio
import numpy as np
from pathlib import Path

def find_ref_patients_xls_line(main_dir, step_name):
    """
    Looks up Excel mapping parameters (sheet name, columns) for a given step.
    
    Returns:
        begin_xls, end_xls, xls_sheet, nb_cell
    """
    # Nettoyage de main_dir
    base_dir = Path(main_dir.lstrip(':')).resolve()

    # Construction du chemin vers le fichier .mat
    ref_path = base_dir / 'Ref_Files' / 'ref_patients_xls.mat'

    # Vérification de sécurité
    if not ref_path.exists():
        print(f"ERREUR : Le fichier est introuvable ici -> {ref_path.absolute()}")
        # Optionnel : Lister les fichiers présents pour voir l'erreur
        if ref_path.parent.exists():
            print(f"Fichiers trouvés dans Ref_Files : {os.listdir(ref_path.parent)}")
        else:
            print("Le dossier 'Ref_Files' n'existe même pas à cet endroit.")
    else:
        mat_contents = sio.loadmat(str(ref_path))
        print("Fichier .mat chargé avec succès.")
    
    # MATLAB: ref_patients_xls is likely a cell array
    # In Python, this becomes a NumPy object array
    ref_patients_xls = mat_contents['ref_patients_xls']
    
    xls_sheet = None
    begin_xls = None
    end_xls = None
    nb_cell = None
    
    # 2. Search for the step_name
    for row in ref_patients_xls:
        # row[0] is step_name_temp
        # Note: .item() or str() is used because scipy often loads 
        # strings as numpy arrays/objects
        step_name_temp = str(row[0][0]) if isinstance(row[0], (list, np.ndarray)) else str(row[0])
        
        if step_name_temp == step_name:
            xls_sheet = str(row[1][0]) if isinstance(row[1], (list, np.ndarray)) else str(row[1])
            begin_xls = str(row[2][0]) if isinstance(row[2], (list, np.ndarray)) else str(row[2])
            end_xls   = str(row[3][0]) if isinstance(row[3], (list, np.ndarray)) else str(row[3])
            nb_cell   = row[4][0] if isinstance(row[4], (list, np.ndarray)) else row[4]
            break # Found it, stop searching

    if xls_sheet is None:
        raise ValueError(f"Step name '{step_name}' not found in ref_patients_xls.mat")

    return begin_xls, end_xls, xls_sheet, nb_cell
=======
import os
import scipy.io as sio
import numpy as np
from pathlib import Path

def find_ref_patients_xls_line(main_dir, step_name):
    """
    Looks up Excel mapping parameters (sheet name, columns) for a given step.
    
    Returns:
        begin_xls, end_xls, xls_sheet, nb_cell
    """
    # Nettoyage de main_dir
    base_dir = Path(main_dir.lstrip(':')).resolve()

    # Construction du chemin vers le fichier .mat
    ref_path = base_dir / 'Ref_Files' / 'ref_patients_xls.mat'

    # Vérification de sécurité
    if not ref_path.exists():
        print(f"ERREUR : Le fichier est introuvable ici -> {ref_path.absolute()}")
        # Optionnel : Lister les fichiers présents pour voir l'erreur
        if ref_path.parent.exists():
            print(f"Fichiers trouvés dans Ref_Files : {os.listdir(ref_path.parent)}")
        else:
            print("Le dossier 'Ref_Files' n'existe même pas à cet endroit.")
    else:
        mat_contents = sio.loadmat(str(ref_path))
        print("Fichier .mat chargé avec succès.")
    
    # MATLAB: ref_patients_xls is likely a cell array
    # In Python, this becomes a NumPy object array
    ref_patients_xls = mat_contents['ref_patients_xls']
    
    xls_sheet = None
    begin_xls = None
    end_xls = None
    nb_cell = None
    
    # 2. Search for the step_name
    for row in ref_patients_xls:
        # row[0] is step_name_temp
        # Note: .item() or str() is used because scipy often loads 
        # strings as numpy arrays/objects
        step_name_temp = str(row[0][0]) if isinstance(row[0], (list, np.ndarray)) else str(row[0])
        
        if step_name_temp == step_name:
            xls_sheet = str(row[1][0]) if isinstance(row[1], (list, np.ndarray)) else str(row[1])
            begin_xls = str(row[2][0]) if isinstance(row[2], (list, np.ndarray)) else str(row[2])
            end_xls   = str(row[3][0]) if isinstance(row[3], (list, np.ndarray)) else str(row[3])
            nb_cell   = row[4][0] if isinstance(row[4], (list, np.ndarray)) else row[4]
            break # Found it, stop searching

    if xls_sheet is None:
        raise ValueError(f"Step name '{step_name}' not found in ref_patients_xls.mat")

    return begin_xls, end_xls, xls_sheet, nb_cell
>>>>>>> theirs
