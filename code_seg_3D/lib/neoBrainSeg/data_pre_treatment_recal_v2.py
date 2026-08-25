import sys
import scipy.io as sio
from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QApplication
from lib.neoBrainSeg.data_pre_treatment_recal_2_v2 import data_pre_treatment_recal_2_v2

def data_pre_treatment_recal_v2(d, idx, main_dir, num_patient, ref):
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Construction des chemins
    # Note : Path gère mieux les chemins absolus sans lstrip(':')
    base_path = Path(main_dir).resolve() / 'Pre_traitement_echo_v2'
    recalage_dir = base_path / 'Recalage' / str(ref)
    repos_dir = base_path / 'cropping' / str(ref)

    path_recal = recalage_dir / f"data_recal_{ref}_d_{d}.mat"
    path_repos = repos_dir / f"data_repos_{ref}.mat"

    should_process = not path_recal.exists()

    if path_recal.exists():
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Recalage")
        msg_box.setText(f"L'examen {ref} a déjà été recalé.\nVoulez-vous recaler de nouveau ?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        reply = msg_box.exec()
        if reply == QMessageBox.Yes:
            should_process = True
        
        msg_box.deleteLater() # Nettoyage mémoire
        app.processEvents()
    
    if should_process:
        if not path_repos.exists():
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Erreur")
            error_box.setText(f"Le fichier de cropping est manquant pour {ref}")
            error_box.exec()
            error_box.deleteLater()
            return

        print(f"Début du recalage de l'examen {ref}")
        try:
            # Chargement des données
            data_repos = sio.loadmat(str(path_repos))['data_repos']
            import numpy as np
            print(f"Data loaded for {ref}, shape: {data_repos.shape}, dtype: {data_repos.dtype}", "min_:" , np.min(data_repos), "max_:", np.max(data_repos))
            
            app.processEvents()
            # Appel de la fonction de calcul
            data_pre_treatment_recal_2_v2(data_repos, d, idx, main_dir, num_patient, ref)
            
            print(f"Examen {ref} recalé avec succès")
            
        except Exception as e:
            print(f"Erreur lors du recalage : {e}")
        finally:
            app.processEvents()