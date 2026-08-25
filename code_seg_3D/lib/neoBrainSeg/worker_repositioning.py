from PySide6.QtCore import QThread, Signal
from lib.neoBrainSeg import database_reader_v2
from lib.neoBrainSeg import dicomfield_extraction
from lib.neoBrainSeg import data_repositioning_v2
from lib.neoBrainSeg import save_repos_data

class RepositioningWorker(QThread):
    """Worker thread gérant le traitement lourd en arrière-plan."""
    
    progress_updated = Signal(int, str)
    finished = Signal()
    error_occurred = Signal(str)
    data_ready_for_viewer = Signal(object) # Envoie les données à la GUI
    
    def __init__(self, acq_dir, idx, main_dir, num_patient, ref, ref_seq_dyn, ref_img_sag, mode_debug):
        super().__init__()
        self.acq_dir = acq_dir
        self.idx = idx
        self.main_dir = main_dir
        self.num_patient = num_patient
        self.ref = ref
        self.ref_seq_dyn = ref_seq_dyn
        self.ref_img_sag = ref_img_sag
        self.mode_debug = mode_debug
    
    def run(self):
        try:
            # Chargement
            self.progress_updated.emit(10, "Chargement des données...")
            data_sag, data_seqdyn = database_reader_v2.database_reader_v2(
                self.acq_dir, self.main_dir, self.ref, self.ref_seq_dyn, self.ref_img_sag
            )
            
            # Dicom Fields
            self.progress_updated.emit(25, "Extraction Dicom...")
            dicom_fields = dicomfield_extraction.dicomfield_extraction(
                self.main_dir, self.ref, self.ref_img_sag, self.ref_seq_dyn
            )
            
            # Repositionnement
            self.progress_updated.emit(50, "Cropping en cours...")
            data_sag_repos, data_seqdyn_repos = data_repositioning_v2.data_repositioning_v2(
                data_sag, data_seqdyn, *dicom_fields[2:], self.ref
            )

            # Sauvegarde 
            self.progress_updated.emit(75, "Enregistrement...")
            save_repos_data.save_repos_data(
                data_sag_repos, data_seqdyn_repos, dicom_fields[0], dicom_fields[1],
                self.idx, self.main_dir, self.num_patient, self.ref, self.mode_debug
            )
            
            # Succès : On envoie la donnée finale
            self.progress_updated.emit(100, "Terminé")
            self.data_ready_for_viewer.emit(data_seqdyn_repos)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()