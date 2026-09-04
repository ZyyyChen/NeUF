<<<<<<< ours
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import Qt
from lib.neoBrainSeg import worker_repositioning
from tools.MITKviewer import valider_recalage

_worker_thread = None

def data_pre_treatment_repositioning_2_v2(acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug):
    global _worker_thread
    
    progress = QProgressDialog(f"Cropping de {ref}...", "Annuler", 0, 100)
    progress.setWindowModality(Qt.ApplicationModal)
    
    _worker_thread = worker_repositioning.RepositioningWorker(
        acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug
    )
    
    # Conteneur pour stocker la donnée du thread
    shared_result = []

    # Connexions
    _worker_thread.progress_updated.connect(lambda val, text: (progress.setValue(val), progress.setLabelText(text)))
    _worker_thread.data_ready_for_viewer.connect(shared_result.append)
    _worker_thread.finished.connect(progress.accept)
    _worker_thread.error_occurred.connect(lambda msg: (print(f"Error: {msg}"), progress.reject()))
    
    progress.canceled.connect(_worker_thread.terminate) 

    _worker_thread.start()
    status = progress.exec() # Bloque ici l'UI proprement

    # Une fois la barre fermée, on lance le viewer sur le thread principal
    if status == QProgressDialog.Accepted and shared_result:
        valider_recalage(shared_result[0])

    # Nettoyage
    if _worker_thread and _worker_thread.isRunning():
        _worker_thread.quit()
        _worker_thread.wait()
    _worker_thread = None
=======
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import Qt
from lib.neoBrainSeg import worker_repositioning
from tools.MITKviewer import valider_recalage

_worker_thread = None

def data_pre_treatment_repositioning_2_v2(acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug):
    global _worker_thread
    
    progress = QProgressDialog(f"Cropping de {ref}...", "Annuler", 0, 100)
    progress.setWindowModality(Qt.ApplicationModal)
    
    _worker_thread = worker_repositioning.RepositioningWorker(
        acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug
    )
    
    # Conteneur pour stocker la donnée du thread
    shared_result = []

    # Connexions
    _worker_thread.progress_updated.connect(lambda val, text: (progress.setValue(val), progress.setLabelText(text)))
    _worker_thread.data_ready_for_viewer.connect(shared_result.append)
    _worker_thread.finished.connect(progress.accept)
    _worker_thread.error_occurred.connect(lambda msg: (print(f"Error: {msg}"), progress.reject()))
    
    progress.canceled.connect(_worker_thread.terminate) 

    _worker_thread.start()
    status = progress.exec() # Bloque ici l'UI proprement

    # Une fois la barre fermée, on lance le viewer sur le thread principal
    if status == QProgressDialog.Accepted and shared_result:
        valider_recalage(shared_result[0])

    # Nettoyage
    if _worker_thread and _worker_thread.isRunning():
        _worker_thread.quit()
        _worker_thread.wait()
    _worker_thread = None
>>>>>>> theirs
