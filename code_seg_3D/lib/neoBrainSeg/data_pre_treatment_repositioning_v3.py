<<<<<<< ours
import pathlib
from PySide6.QtWidgets import QMessageBox
# Assuming this is inside lib/neoBrainSeg/, we import the next step
from lib.neoBrainSeg.data_pre_treatment_repositioning_2_v2 import data_pre_treatment_repositioning_2_v2

def data_pre_treatment_repositioning_v3(acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug):
    """
    Checks for existing repositioning data and prompts user for overwrite if found.
    """
    
    # Define the path using pathlib (cleaner than fullfile)
    # main_dir / 'folder' / 'subfolder' automatically adds correct slashes
    base_path = pathlib.Path(main_dir)
    file_path = (base_path / 'Pre_traitement_echo_v2' / 'Cropping' / 
                 str(ref) / f"data_repos_{ref}.mat")

    run_step = False

    # Equivalent to if ~exist(...)
    if not file_path.exists():
        run_step = True
        #raise FileExistsError(f'{file_path} not found')
    
    # Equivalent to elseif exist(...)
    else:
        # Create the PySide6 Question Dialog (questdlg)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle('Cropping existant')
        msg_box.setText(f"Voulez effectuer de nouveau le cropping de {ref}?")
        
        # Add buttons and set 'Non' as default
        oui_button = msg_box.addButton("Oui", QMessageBox.YesRole)
        non_button = msg_box.addButton("Non", QMessageBox.NoRole)
        msg_box.setDefaultButton(non_button)
        
        msg_box.exec()

        if msg_box.clickedButton() == oui_button:
            run_step = True

    # Call the secondary function if logic dictates
    if run_step:
        data_pre_treatment_repositioning_2_v2(
            acq_dir, idx, main_dir, num_patient, 
            ref, ref_seq_dyn, ref_img_sag, mode_debug
        )
=======
import pathlib
from PySide6.QtWidgets import QMessageBox
# Assuming this is inside lib/neoBrainSeg/, we import the next step
from lib.neoBrainSeg.data_pre_treatment_repositioning_2_v2 import data_pre_treatment_repositioning_2_v2

def data_pre_treatment_repositioning_v3(acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug):
    """
    Checks for existing repositioning data and prompts user for overwrite if found.
    """
    
    # Define the path using pathlib (cleaner than fullfile)
    # main_dir / 'folder' / 'subfolder' automatically adds correct slashes
    base_path = pathlib.Path(main_dir)
    file_path = (base_path / 'Pre_traitement_echo_v2' / 'Cropping' / 
                 str(ref) / f"data_repos_{ref}.mat")

    run_step = False

    # Equivalent to if ~exist(...)
    if not file_path.exists():
        run_step = True
        #raise FileExistsError(f'{file_path} not found')
    
    # Equivalent to elseif exist(...)
    else:
        # Create the PySide6 Question Dialog (questdlg)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle('Cropping existant')
        msg_box.setText(f"Voulez effectuer de nouveau le cropping de {ref}?")
        
        # Add buttons and set 'Non' as default
        oui_button = msg_box.addButton("Oui", QMessageBox.YesRole)
        non_button = msg_box.addButton("Non", QMessageBox.NoRole)
        msg_box.setDefaultButton(non_button)
        
        msg_box.exec()

        if msg_box.clickedButton() == oui_button:
            run_step = True

    # Call the secondary function if logic dictates
    if run_step:
        data_pre_treatment_repositioning_2_v2(
            acq_dir, idx, main_dir, num_patient, 
            ref, ref_seq_dyn, ref_img_sag, mode_debug
        )
>>>>>>> theirs
