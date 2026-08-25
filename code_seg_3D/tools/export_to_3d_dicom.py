import SimpleITK as sitk
import os
import time
import numpy as np

def export_to_3d_dicom(data, output_filepath, patient_name, patient_id, series_name="NeoBrain_3D_Vol"):
    """
    Sauvegarde un volume 3D en un fichier DICOM unique Multi-frame (8-bit).
    Incorpore les tags de contraste et d'orientation pour éviter les vues blanches.
    """
    # 1. Sécurité : Forcer l'extension .dcm
    if not output_filepath.lower().endswith('.dcm'):
        output_filepath += ".dcm"

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Prétraitement des données (Conversion en uint8 0-255)
    # DICOM 8-bit nécessite du non-signé (uint8) pour une compatibilité maximale
    data = data.transpose(0, 2, 1)
    data = np.flip(data, axis=0)
    if data.dtype != np.uint8:
        d_max = np.max(data)
        if d_max > 0:
            # Si déjà normalisé (0-1.01) ou en 0-127, on scale à 0-255
            if d_max <= 1.01:
                data = (data * 255).astype(np.uint8)
            elif d_max <= 127:
                data = (data.astype(float) * (255/127)).astype(np.uint8)
            else:
                data = (255 * (data / d_max)).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

    # 3. Récupération automatique du spacing depuis le fichier
   
    dx, dy, dz = (1.0, 1.0, 1.0)

    # 4. Création de l'image SimpleITK
    # Note : SimpleITK attend l'ordre (Z, Y, X)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing([float(dx), float(dy), float(dz)])
    image.SetOrigin([0.0, 0.0, 0.0])

    # 5. Configuration des Métadonnées (Tags DICOM)
    mod_time = time.strftime("%H%M%S")
    mod_date = time.strftime("%Y%m%d")

    # --- NOUVEAU : Identité du Patient ---
    image.SetMetaData("0010|0010", patient_name)  # Patient's Name
    image.SetMetaData("0010|0020", patient_id)    # Patient ID

    # --- Tags Display & Contraste (Crucial pour éviter l'écran blanc) ---
    image.SetMetaData("0028|0004", "MONOCHROME2") # Interprétation photométrique
    image.SetMetaData("0028|1050", "128")         # Window Center (Luminosité)
    image.SetMetaData("0028|1051", "256")         # Window Width (Contraste)

    # --- Tags de Géométrie 3D (Permet la reconstruction MPR) ---
    # Orientation axiale standard : définit les axes pour reconstruire Sagittal/Coronal
    image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0") 
    
    # --- Tags de Stockage 8-bit ---
    image.SetMetaData("0028|0100", "8")  # Bits Allocated
    image.SetMetaData("0028|0101", "8")  # Bits Stored
    image.SetMetaData("0028|0102", "7")  # High Bit
    image.SetMetaData("0028|0103", "0")  # Pixel Representation (0 = Unsigned)

    # --- Tags Système & Multi-frame ---
    image.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.7.4") # Multi-frame Secondary Capture
    image.SetMetaData("0028|0008", str(data.shape[0]))            # Nombre de coupes
    image.SetMetaData("0028|0030", f"{dx}\\{dy}")                 # Spacing XY
    image.SetMetaData("0018|0050", str(dz))                       # Épaisseur Z
    image.SetMetaData("0008|103e", series_name)
    image.SetMetaData("0008|0021", mod_date)
    image.SetMetaData("0008|0031", mod_time)

    # 6. Exécution de l'écriture
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_filepath)
        writer.Execute(image)
        print(f"Succès : DICOM 3D unique sauvegardé sous {output_filepath}")
    except Exception as e:
        print(f"Erreur lors de l'écriture DICOM : {e}")
        raise

# --- Exemple d'utilisation dans votre boucle ---
# max_v = np.max(self.data_repcom)
# data_8bit = (self.data_repcom / max_v * 255).astype(np.uint8) if max_v > 0 else self.data_repcom.astype(np.uint8)
# mhd_ref = os.path.join(output_path, f'data_repcom_{self.ref}_mitk.mhd')
# out_dcm = os.path.join(output_path, f'data_repcom_{self.ref}.dcm')
# export_to_3d_dicom_8bit(data_8bit, out_dcm, mhd_ref)