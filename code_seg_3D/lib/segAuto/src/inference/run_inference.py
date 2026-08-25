import os
import sys
import shutil
import logging
from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import Qt

# Import de vos modules spécifiques (assurez-vous que les chemins sont corrects)
from lib.segAuto.src.inference.Vnet_inference import SetVolumePath, LoadVolume, Inference

# Configuration TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les logs TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logger = logging.getLogger(__name__)

def run(dataset_path, maindir, ref):
    """
    Fonction principale d'inférence.
    Affiche une popup si le résultat existe déjà.
    """
    
    # 1. Gestion de l'instance QApplication (Indispensable pour la GUI)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 2. Vérification de l'existence des fichiers de sortie
    output_dir = Path(maindir) / "SegAuto_VNet" / str(ref)
    seg_vol_file = output_dir / f'{ref}_auto_seg_1.nii.gz'
    
    should_process = True # Par défaut, on traite

    if seg_vol_file.exists():
        # Juste avant le "diag = QMessageBox()"
        app.processEvents()
        QApplication.processEvents()
        # Création de la boîte de dialogue
        diag = QMessageBox()
        # Garanti que la fenêtre soit visible et au-dessus
        diag.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog) 
        diag.setWindowTitle('Fichier existant')
        diag.setText(f"La segmentation de {ref} existe déjà.")
        diag.setInformativeText("Voulez-vous écraser les fichiers et relancer l'inférence ?")
        diag.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        diag.setDefaultButton(QMessageBox.No)
        
        # Forcer le focus
        diag.raise_()
        diag.activateWindow()
        
        # Exécution et blocage jusqu'à réponse
        result = diag.exec() 
        
        if result == QMessageBox.No:
            print(f"-> Segment {ref} déjà présent. Passage au suivant.")
            return  # On arrête la fonction ici proprement
        else:
            print(f"-> Écrasement autorisé pour {ref}.")
            should_process = True

    # 3. Lancement du processus de traitement
    if should_process:
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            print(f"[ERROR] Le dossier source n'existe pas : {dataset_dir}")
            return

        # Liste des fichiers DICOM (exclure les .mat de Matlab)
        dcm_files = [f for f in os.listdir(dataset_dir) if not f.endswith('.mat')]
        
        if not dcm_files:
            print(f"[ERROR] Aucun fichier DICOM trouvé dans {dataset_dir}")
            return

        # Sélection du premier volume
        volume_path = dataset_dir / dcm_files[0]
        
        if volume_path.is_file() and volume_path.suffix.lower() == '.dcm':
            data_name = volume_path.name
            print(f"Sequence found: {data_name}")
            
            try:
                # Préparation des données pour V-Net
                SetVolumePath(str(volume_path), data_name)
                vol, label = LoadVolume(maindir, volume_path)
                
                print(f"Volume chargé: {vol[0].shape}")
                
                # --- Lancement de l'Inférence ---
                print("Lancement de l'inférence V-Net...")
                Inference(maindir, ref)
                print("[SUCCESS] Inférence terminée.")

                # 4. Nettoyage des dossiers temporaires après succès
                temp_dirs = [
                    Path(maindir) / "newold_dataset_640_resize_320",
                    Path(maindir) / "logdir"
                ]
                
                for d in temp_dirs:
                    if d.exists() and d.is_dir():
                        shutil.rmtree(d)
                        print(f"Nettoyage : {d.name} supprimé.")

            except Exception as e:
                print(f"[CRITICAL ERROR] Échec de l'inférence : {e}")
                raise e
        else:
            print(f"[ERROR] Le fichier {volume_path} n'est pas un DICOM valide.")

# Bloc de test si le script est lancé seul
if __name__ == '__main__':
    # Exemple de test (ajustez les chemins pour vos tests locaux)
    # run("C:/Data/Test", "C:/Project/Main", "Patient01")
    pass