# Import des sous-fonctions de votre librairie
from lib.neoBrainSeg.data_registration_v2 import data_registration_v2
from lib.neoBrainSeg.save_registered_data import save_registered_data
from PySide6.QtWidgets import QApplication
import sys

def data_pre_treatment_recal_2_v2(data_repos, d, idx, main_dir, num_patient, ref):
    """
    Recalage optimisé avec gestion de flux et mémoire.
    """
    app = QApplication.instance()

    # 1. Recalage de la séquence
    # C'est ici que le CPU/GPU travaille le plus.
    print(f"   Exécution du recalage (data_registration_v2) pour {ref}...")
    data_recal, t_form = data_registration_v2(d, data_repos)
    
    # Libération immédiate de la mémoire de l'étape précédente
    del data_repos 
    
    # Petite pause pour laisser l'interface respirer après le gros calcul
    if app:
        app.processEvents()

    # 2. Sauvegarde des résultats
    print(f"   Sauvegarde des résultats pour {ref}...")
    try:
        save_registered_data(d, data_recal, main_dir, num_patient, ref, t_form)
        print(f"   [OK] Données sauvegardées avec succès.")
    except Exception as e:
        print(f"   [ERREUR] Échec de la sauvegarde : {e}")
        raise # On propage l'erreur pour que le wrapper v3 sache que ça a échoué

    if app:
        app.processEvents()

    return data_recal # Optionnel : permet d'enchaîner sans recharger le .mat
