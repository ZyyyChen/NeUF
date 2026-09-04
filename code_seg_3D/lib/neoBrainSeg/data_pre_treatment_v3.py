<<<<<<< ours
"""Python translation of the MATLAB `data_pre_treatment_v3` wrapper (package version).

Optimized for PySide6 event loop stability and memory management.
"""

import logging
import sys
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optionnel : désactive explicitement oneDNN

from typing import Sequence, Any
from PySide6.QtWidgets import QApplication
from lib.neoBrainSeg.ref_parameters_extraction_v3 import ref_parameters_extraction_v3
from lib.neoBrainSeg.data_pre_treatment_repositioning_v3 import data_pre_treatment_repositioning_v3
from lib.neoBrainSeg.data_pre_treatment_recal_v2 import data_pre_treatment_recal_v2
from lib.neoBrainSeg.recons3D import data_pre_treatment_recons3D_v2
from lib.neoBrainSeg.Repere_commun import data_pre_treatment_repcom_v2
from lib.neoBrainSeg.check_ref_patients_echo_v3 import check_ref_patients_update
from lib.segAuto.src.inference.run_inference import run

logger = logging.getLogger(__name__)

# --- CONFIGURATION UNIQUE DE L'APPLICATION ---
app = QApplication.instance() or QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

def refresh_gui():
    """Traite les événements GUI en attente et force la libération mémoire."""
    if QApplication.instance():
        QApplication.instance().processEvents()
    gc.collect()

def data_pre_treatment_v3(
    active_step_data: Sequence[int],
    d: float,
    debug_mode: bool,
    idx: int,
    main_dir: str,
    ref_patients_echo: Any,
    params_patients_echo_v2: Any,
    T: float,
) -> None:
    """Run the preprocessing pipeline according to `active_step_data` range."""
    
    # Validation des entrées
   
    active_step_list = list(active_step_data)
    min_step, max_step = active_step_list[0], active_step_list[-1]
    

    if not isinstance(idx, int): raise TypeError("idx doit être un entier")

    logger.info(f"--- Démarrage Pipeline: Patient No {idx+1} in the excel file | Steps {min_step}-{max_step} ---")
    
    try:
        # Extraction des paramètres (Acquisition & Références)
        params = ref_parameters_extraction_v3(idx, params_patients_echo_v2, ref_patients_echo)
        
        # Mapping explicite des variables nécessaires
        (acq_dir, _, angle_rot_cor, angle_rot_sag, 
         delta_X_cc, delta_X_seqdyn, num_patient, origin_coord, ref, 
         ref_seq_dyn, ref_img_sag) = params

        step = 1
        # 1. Repositionnement (Cropping)
        if step in active_step_list:
            logger.info(f"Step {step}: Cropping")
            data_pre_treatment_repositioning_v3(
                acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, debug_mode
            )
            refresh_gui()

        # update param
        params_v2 = check_ref_patients_update(main_dir, idx)
        delta_X_seqdyn, delta_X_cc = params_v2[0][-2], params_v2[0][-1]
        print("UPDATE : delta_X_seqdyn", delta_X_seqdyn, "delta_X_cc", delta_X_cc)

        # 2. Recalage
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Recalage")
            data_pre_treatment_recal_v2(d, idx, main_dir, num_patient, ref)
            refresh_gui()

        # 3. Reconstruction 3D
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Reconstruction 3D")
            data_pre_treatment_recons3D_v2(
                d, delta_X_cc, delta_X_seqdyn, debug_mode, idx, main_dir, num_patient, ref
            )
            refresh_gui()

        # 4. Repère Commun
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Passage en repère commun")
            data_pre_treatment_repcom_v2(
                angle_rot_cor, angle_rot_sag, debug_mode, idx, main_dir, 
                num_patient, ref, T, origin_coord
            )
            refresh_gui()
        
        # 5. Segmentation Automatique
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Segmentation Automatique")
            # Petit message propre pour savoir que ça tourne
            print("\n" + "="*50)
            print(" INFERENCE ENGINE - V-NET TENSORFLOW ")
            print("="*50)
            
            try:
                run(dataset_path=os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Repere_commun', str(ref)), maindir=main_dir, ref=ref)
                print("\n[SUCCESS] Inférence terminée.")
            except Exception as e:
                print(f"\n[ERROR] Une erreur est survenue : {e}")
            refresh_gui()

    except Exception as e:
        logger.error(f"Erreur critique lors du traitement du patient {idx+1}: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Nettoyage final pour le patient {idx+1}")
        refresh_gui()
=======
"""Python translation of the MATLAB `data_pre_treatment_v3` wrapper (package version).

Optimized for PySide6 event loop stability and memory management.
"""

import logging
import sys
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les messages INFO et WARNING de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Optionnel : désactive explicitement oneDNN

from typing import Sequence, Any
from PySide6.QtWidgets import QApplication
from lib.neoBrainSeg.ref_parameters_extraction_v3 import ref_parameters_extraction_v3
from lib.neoBrainSeg.data_pre_treatment_repositioning_v3 import data_pre_treatment_repositioning_v3
from lib.neoBrainSeg.data_pre_treatment_recal_v2 import data_pre_treatment_recal_v2
from lib.neoBrainSeg.recons3D import data_pre_treatment_recons3D_v2
from lib.neoBrainSeg.Repere_commun import data_pre_treatment_repcom_v2
from lib.neoBrainSeg.check_ref_patients_echo_v3 import check_ref_patients_update
from lib.segAuto.src.inference.run_inference import run

logger = logging.getLogger(__name__)

# --- CONFIGURATION UNIQUE DE L'APPLICATION ---
app = QApplication.instance() or QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

def refresh_gui():
    """Traite les événements GUI en attente et force la libération mémoire."""
    if QApplication.instance():
        QApplication.instance().processEvents()
    gc.collect()

def data_pre_treatment_v3(
    active_step_data: Sequence[int],
    d: float,
    debug_mode: bool,
    idx: int,
    main_dir: str,
    ref_patients_echo: Any,
    params_patients_echo_v2: Any,
    T: float,
) -> None:
    """Run the preprocessing pipeline according to `active_step_data` range."""
    
    # Validation des entrées
   
    active_step_list = list(active_step_data)
    min_step, max_step = active_step_list[0], active_step_list[-1]
    

    if not isinstance(idx, int): raise TypeError("idx doit être un entier")

    logger.info(f"--- Démarrage Pipeline: Patient No {idx+1} in the excel file | Steps {min_step}-{max_step} ---")
    
    try:
        # Extraction des paramètres (Acquisition & Références)
        params = ref_parameters_extraction_v3(idx, params_patients_echo_v2, ref_patients_echo)
        
        # Mapping explicite des variables nécessaires
        (acq_dir, _, angle_rot_cor, angle_rot_sag, 
         delta_X_cc, delta_X_seqdyn, num_patient, origin_coord, ref, 
         ref_seq_dyn, ref_img_sag) = params

        step = 1
        # 1. Repositionnement (Cropping)
        if step in active_step_list:
            logger.info(f"Step {step}: Cropping")
            data_pre_treatment_repositioning_v3(
                acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, debug_mode
            )
            refresh_gui()

        # update param
        params_v2 = check_ref_patients_update(main_dir, idx)
        delta_X_seqdyn, delta_X_cc = params_v2[0][-2], params_v2[0][-1]
        print("UPDATE : delta_X_seqdyn", delta_X_seqdyn, "delta_X_cc", delta_X_cc)

        # 2. Recalage
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Recalage")
            data_pre_treatment_recal_v2(d, idx, main_dir, num_patient, ref)
            refresh_gui()

        # 3. Reconstruction 3D
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Reconstruction 3D")
            data_pre_treatment_recons3D_v2(
                d, delta_X_cc, delta_X_seqdyn, debug_mode, idx, main_dir, num_patient, ref
            )
            refresh_gui()

        # 4. Repère Commun
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Passage en repère commun")
            data_pre_treatment_repcom_v2(
                angle_rot_cor, angle_rot_sag, debug_mode, idx, main_dir, 
                num_patient, ref, T, origin_coord
            )
            refresh_gui()
        
        # 5. Segmentation Automatique
        step += 1
        #if min_step <= step <= max_step:
        if step in active_step_list:
            logger.info(f"Step {step}: Segmentation Automatique")
            # Petit message propre pour savoir que ça tourne
            print("\n" + "="*50)
            print(" INFERENCE ENGINE - V-NET TENSORFLOW ")
            print("="*50)
            
            try:
                run(dataset_path=os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Repere_commun', str(ref)), maindir=main_dir, ref=ref)
                print("\n[SUCCESS] Inférence terminée.")
            except Exception as e:
                print(f"\n[ERROR] Une erreur est survenue : {e}")
            refresh_gui()

    except Exception as e:
        logger.error(f"Erreur critique lors du traitement du patient {idx+1}: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Nettoyage final pour le patient {idx+1}")
        refresh_gui()
>>>>>>> theirs
