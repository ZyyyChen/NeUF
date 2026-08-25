import sys
import os
import cv2
import math
import logging
import numpy as np
import scipy.io as sio
import openpyxl
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PySide6.QtCore import Qt, QThread, Signal
from scipy import ndimage
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
from tools.export_to_3d_dicom import export_to_3d_dicom

# Imports locaux supposés existants
from tools.gui_lay3d import gui_lay3d
from tools.MITKviewer import *

logger = logging.getLogger(__name__)

# --- INITIALISATION UNIQUE DE L'APPLICATION ---
app = QApplication.instance() or QApplication(sys.argv)

from pathlib import Path
from PySide6 import QtCore, QtWidgets

def choose_recons3D_origin_v2(data_3D, 
                              origin_coord=None, 
                              im_demo_path_sag=next(Path(os.path.abspath(__file__)).parents[3].rglob("CoordonnéesSag_Splenium_CorpsCalleux.png"), None),
                              im_demo_path_cor=next(Path(os.path.abspath(__file__)).parents[3].rglob("CoordonnéesCoro_Splenium_CorpsCalleux.png"), None)
                              ):
    
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # --- 1. CONFIGURATION INITIALE (ANTI-CLIGNOTEMENT) ---
    plt.ioff() # Désactive l'interactivité automatique
    
    # Fenêtre Interaction (h)
    # Note: On suppose que gui_lay3d renvoie (fig, fx, fy, fz, sld, bp, bn, switch_mode)
    h, fx, fy, fz, sld, bp, bn, switch_mode = gui_lay3d(data_3D)
    main_win = h.canvas.manager.window
    main_win.setGeometry(620, 100, 950, 750)
    main_win.setWindowFlags(main_win.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    
    # Fenêtre Aide (999)
    fig_help = plt.figure(999, figsize=(5, 5))
    help_win = fig_help.canvas.manager.window
    help_win.setGeometry(50, 100, 550, 550)
    help_win.setWindowFlags(help_win.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

    # Affichage des fenêtres
    main_win.show()
    help_win.show()

    state = {"clicked": False, "x": 0.0, "y": 0.0}

    def onclick(event):
        """Gère le clic et dessine un point rouge unique."""
        if event.inaxes is None or event.inaxes in [sld.ax, bp.ax, bn.ax]: 
            return
        
        state["x"], state["y"] = event.xdata, event.ydata
        
        # Nettoyage des points existants sur l'axe
        if hasattr(event.inaxes, 'lines'):
            while len(event.inaxes.lines) > 0:
                event.inaxes.lines[0].remove()
        
        event.inaxes.plot(event.xdata, event.ydata, 'r+', markersize=12, markeredgewidth=2)
        h.canvas.draw_idle()
        state["clicked"] = True

    h.canvas.mpl_connect('button_press_event', onclick)

    def wait_for_click():
        """Boucle d'attente stable."""
        state["clicked"] = False
        while not state["clicked"]:
            app.processEvents() 
            if not plt.fignum_exists(h.number): return False
            QtCore.QThread.msleep(20) # Crucial pour éviter le scintillement
        return True

    def clear_all_axes():
        """Efface tous les points tracés sur la figure principale."""
        for ax in h.axes:
            if ax not in [sld.ax, bp.ax, bn.ax]:
                while len(ax.lines) > 0:
                    ax.lines[0].remove()
        h.canvas.draw()

    def get_val_custom(title, label, value):
        """Dialogue de saisie forcé au premier plan."""
        dialog = QtWidgets.QInputDialog(main_win)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        dialog.setDoubleRange(-20000.0, 20000.0)
        dialog.setDoubleDecimals(1)
        dialog.setDoubleValue(float(value))
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        ok = dialog.exec_()
        return dialog.doubleValue(), ok == QtWidgets.QDialog.Accepted

    try:
        # ==========================================
        # PHASE A : SAGITTAL
        # ==========================================
        fig_help.clf()
        ax_h = fig_help.add_subplot(111)
        if im_demo_path_sag and os.path.exists(im_demo_path_sag):
            ax_h.imshow(plt.imread(im_demo_path_sag))
        ax_h.set_title("Example : Coordonnées SAGITTAL \nSplenium CorpsCalleux (point rouge)", fontweight='bold', color='red')
        ax_h.axis('off')
        fig_help.canvas.draw()
        
        switch_mode('Y', 1)
        h.canvas.draw()

        valid_a = False
        while not valid_a:
            print("En attente de clic (Sagittal)...")
            if not wait_for_click(): return None
            
            # --- Votre logique de validation ---
            val_y, ok_y = get_val_custom("Coord Y", "Confirmer coord Y:", state["y"])
            if not ok_y: 
                clear_all_axes() # Efface le point si on annule
                continue 
            
            val_x_s, ok_x_s = get_val_custom("Coord X", "Confirmer coord X:", state["x"])
            if not ok_x_s: 
                clear_all_axes()
                continue 
            
            coord_x, coord_z = val_y, val_x_s
            valid_a = True

        # NETTOYAGE ENTRE LES PHASES
        clear_all_axes()

        # ==========================================
        # PHASE B : CORONAL
        # ==========================================
        fig_help.clf()
        ax_h = fig_help.add_subplot(111)
        if im_demo_path_cor and os.path.exists(im_demo_path_cor):
            ax_h.imshow(plt.imread(im_demo_path_cor))
        ax_h.set_title("Exemple : Coordonnées CORONAL Splenium \nCorps Calleux (point rouge)", fontweight='bold', color='red')
        ax_h.axis('off')
        fig_help.canvas.draw()

        switch_mode('X', 2)
        h.canvas.draw()

        valid_b = False
        while not valid_b:
            print("En attente de clic (Coronal)...")
            if not wait_for_click(): return None
            
            val_x_c, ok_x_c = get_val_custom("Coord X", "Confirmer coord X:", state["x"])
            if not ok_x_c: 
                clear_all_axes()
                continue 
            
            coord_y = val_x_c
            valid_b = True

    except Exception as e:
        print(f"Erreur durant l'exécution : {e}")
        return None
    finally:
        plt.close(999) 
        plt.close(h)   
        
    print(f"Origine validée : X={coord_x:.1f}, Y={coord_y:.1f}, Z={coord_z:.1f}")
    return coord_x, coord_y, coord_z

def data_rotation_v2(angle_rot, data_3D, view2rot):
    
    data_np = np.asarray(data_3D)
    
    # Correction de l'axe de progression selon la vue
    total_steps = data_np.shape[2] if view2rot == 'cor' else data_np.shape[1]
    
    progress = QProgressDialog(f"Rotation {view2rot} ({angle_rot}°)...", "Annuler", 0, total_steps)
    progress.setWindowModality(Qt.WindowModal)
    progress.show()

    def apply_rotation_internal(volume, angle):
        z_dim, y_dim, x_dim = volume.shape
        center = (y_dim // 2, z_dim // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        out = np.zeros_like(volume)
        
        for i in range(x_dim):
            out[:,:,i] = cv2.warpAffine(volume[:,:,i], M, (y_dim, z_dim), flags=cv2.INTER_NEAREST)
            if i % 10 == 0: # Update progress every 10 slices for performance
                progress.setValue(i)
                app.processEvents()
            if progress.wasCanceled(): break
        return out

    if view2rot == 'cor':
        data_3D_rot = apply_rotation_internal(data_np, angle_rot)
    elif view2rot == 'sag':
        data_temp = np.transpose(data_np, (0, 2, 1))
        data_rot_temp = apply_rotation_internal(data_temp, angle_rot)
        data_3D_rot = np.transpose(data_rot_temp, (0, 2, 1))
    else:
        progress.close()
        raise ValueError("view2rot doit être 'cor' ou 'sag'")

    progress.setValue(total_steps)
    progress.close()
    return data_3D_rot

def determine_img_angle_rot_reg_v2(data_3D, view2rot, img_demo_path="demo.png"):
    """Interface avec deux fenêtres occupant chacune une moitié d'écran."""
    plt.ion() 
    
    # 1. Sélection de la coupe
    fig_h, fx, fy, fz, sld, bp, bn, switch_mode = gui_lay3d(data_3D)
    
    if view2rot == 'cor':
        switch_mode('X', 2) 
        angle_min, angle_max = -15, 15
    else:
        switch_mode('Y', 1) 
        angle_min, angle_max = -35, 35

    plt.show()
    plt.pause(0.1) 
    
    ans = input(f"\n--- Sélection de la coupe ({view2rot}) ---\nIndex de coupe [Défaut: {int(sld.val)}] : ")
    img_ref_idx = int(ans) if ans.strip() else int(sld.val)
    plt.close(fig_h)

    # --- CALCUL DES DIMENSIONS DE L'ÉCRAN ---
    # On utilise le manager de Matplotlib pour accéder aux fonctions système
    slice_data = data_3D[:, :, img_ref_idx] if view2rot == 'cor' else data_3D[:, img_ref_idx, :]
    
    # Création des figures
    fig_demo = plt.figure("1. RÉFÉRENCE")
    fig_rot = plt.figure("2. RÉGLAGE")
    
    try:
        # Récupération de la géométrie de l'écran via le backend Qt
        mgr_demo = fig_demo.canvas.manager
        mgr_rot = fig_rot.canvas.manager
        
        # On récupère la taille de l'écran (Desktop)
        screen = mgr_demo.window.screen().availableGeometry()
        screen_w = screen.width()
        screen_h = screen.height()
        
        # Calcul de la moitié (Moitié-Moitié)
        half_w = screen_w // 3
        
        # Redimensionnement et placement (x, y, largeur, hauteur)
        mgr_demo.window.setGeometry(0, 0, half_w, screen_h)
        mgr_rot.window.setGeometry(half_w, 0, 2*half_w, screen_h)
        
        # Forcer le premier plan
        mgr_demo.window.raise_()
        mgr_rot.window.raise_()
        mgr_demo.window.activateWindow()
        mgr_rot.window.activateWindow()
        
    except Exception as e:
        print(f"Note: Placement automatique limité ({e})")

    # --- AFFICHAGE FIGURE DÉMO (À GAUCHE) ---
    plt.figure(fig_demo.number)
    try:
        demo_img = mpimg.imread(img_demo_path)
        plt.imshow(demo_img)
        plt.axis('off')
        plt.title("MODÈLE DE RÉFÉRENCE")
    except:
        plt.text(0.5, 0.5, "Image non trouvée")

    # --- AFFICHAGE FIGURE ROTATION (À DROITE) ---
    plt.figure(fig_rot.number)
    ax_rot = fig_rot.add_subplot(111)
    plt.subplots_adjust(bottom=0.2)
    img_display = ax_rot.imshow(slice_data, cmap='gray', aspect='auto')
    
    # Grille
    h_img, w_img = slice_data.shape
    for x in range(0, w_img, 50): ax_rot.axvline(x, color='lime', lw=0.5, alpha=0.6)
    for y in range(0, h_img, 50): ax_rot.axhline(y, color='lime', lw=0.5, alpha=0.6)

    ax_slider = fig_rot.add_axes([0.25, 0.08, 0.5, 0.04])
    angle_slider = Slider(ax_slider, 'Angle (°)', angle_min, angle_max, valinit=0)

    def update_rot(val):
        rotated = ndimage.rotate(slice_data, val, reshape=False, order=1)
        img_display.set_data(rotated)
        fig_rot.canvas.draw_idle()

    angle_slider.on_changed(update_rot)
    
    plt.show()
    plt.pause(0.5) # Petit temps pour stabiliser l'affichage
    
    # --- BOUCLE D'ATTENTE ---
    print(f"--- Réglage de l'angle ({view2rot}) ---")
    while True:
        plt.pause(0.1)
        ans = input(f"Validez l'angle (Actuel: {angle_slider.val:.2f}°) [Entrée] : ")
        break 

    angle_rot = float(ans) if ans.strip() else angle_slider.val
    plt.close(fig_rot)
    plt.close(fig_demo)
    plt.ioff()
    return angle_rot

# --- Thread interne pour le traitement lourd ---
class _SaveDataWorker(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal()

    def __init__(self, angle_rot_cor, angle_rot_sag, coord_x, coord_y, coord_z,
                 data_repcom, idx, main_dir, min_x, min_y, min_z, num_patient, ref, debug_mode):
        super().__init__()
        self.angle_rot_cor = angle_rot_cor
        self.angle_rot_sag = angle_rot_sag
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z
        self.data_repcom = data_repcom
        self.idx = idx
        self.main_dir = main_dir
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.num_patient = num_patient
        self.ref = ref
        self.debug_mode = debug_mode

    def run(self):
        try:
            # Dossier de sortie
            output_path = os.path.join(self.main_dir, 'Pre_traitement_echo_v2', 'Repere_commun', self.ref)
            os.makedirs(output_path, exist_ok=True)
            self.progress_signal.emit(10)

            # Sauvegarde du fichier .mat principal
            
            mat_filename = os.path.join(output_path, f'data_repcom_{self.ref}.mat')
            sio.savemat(mat_filename, {
                'data_repcom': self.data_repcom,
                'min_x': self.min_x,
                'min_y': self.min_y,
                'min_z': self.min_z
            }, do_compression=True)
            self.progress_signal.emit(30)

            # 3. Export MITK (Vérification de l'existence de la fonction externe)
            try:
                # Scale data to 0-255 for 8-bit
                max_v = np.max(self.data_repcom)
                if max_v > 0:
                    data_8bit = (self.data_repcom / max_v * 255).astype(np.uint8)
                else:
                    data_8bit = self.data_repcom.astype(np.uint8)

                # Export Unique 3D DICOM
                base_name = f'data_repcom_{self.ref}'
                output_dicom_file = os.path.join(output_path, base_name + ".dcm")

                export_to_3d_dicom(data_8bit, output_dicom_file, self.num_patient, self.ref)
            except NameError:
                print("Avertissement: export_rawmhd n'est pas définie.")
            self.progress_signal.emit(60)

            # 4. Mise à jour du fichier Excel
            excel_path = os.path.join(self.main_dir, 'Ref_Files', 'RefPatientsUS.xlsx')
            if os.path.exists(excel_path):
                try:
                    wb = openpyxl.load_workbook(excel_path)
                    if 'ReconstructionParameters' in wb.sheetnames:
                        sheet = wb['ReconstructionParameters']
                        row_idx = self.idx + 3
                        """sheet.cell(row=row_idx, column=10).value = self.coord_x
                        sheet.cell(row=row_idx, column=11).value = self.coord_y
                        sheet.cell(row=row_idx, column=12).value = self.coord_z
                        sheet.cell(row=row_idx, column=13).value = self.angle_rot_cor
                        sheet.cell(row=row_idx, column=14).value = self.angle_rot_sag"""
                        sheet.cell(row=row_idx, column=3).value = self.coord_x
                        sheet.cell(row=row_idx, column=4).value = self.coord_y
                        sheet.cell(row=row_idx, column=5).value = self.coord_z
                        sheet.cell(row=row_idx, column=6).value = self.angle_rot_cor
                        sheet.cell(row=row_idx, column=7).value = self.angle_rot_sag

                        wb.save(excel_path)
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde Excel (le fichier est peut-être ouvert) : {e}")
            self.progress_signal.emit(80)

            # 5. Mise à jour des paramètres globaux .mat
            if self.debug_mode:

                param_path = os.path.join(self.main_dir, 'Ref_Files', 'ReconstructionParameters_v2.mat')
                if os.path.exists(param_path):
                    params_file = sio.loadmat(param_path)
                    if 'params_patients_echo_v2' in params_file:
                        params_patients = params_file['params_patients_echo_v2']
                        params_patients[self.idx+1, 7] = self.coord_x
                        params_patients[self.idx+1, 8] = self.coord_y
                        params_patients[self.idx+1, 9] = self.coord_z
                        params_patients[self.idx+1, 10] = self.angle_rot_cor
                        params_patients[self.idx+1, 11] = self.angle_rot_sag
                        sio.savemat(param_path, {'params_patients_echo_v2': params_patients})
                else:
                    raise FileNotFoundError
            
            self.progress_signal.emit(100)

        except Exception as e:
            print(f"Erreur critique dans le thread : {e}")
        finally:
            self.finished_signal.emit()

# --- Fonction de haut niveau pour appeler la sauvegarde ---
def save_data_repcom_v2(angle_rot_cor, angle_rot_sag, coord_x, coord_y, coord_z,
                        data_repcom, idx, main_dir, min_x, min_y, min_z, num_patient, ref, debug_mode):
    
    app = QApplication.instance() or QApplication([])

    # Configuration de la barre de progression
    progress = QProgressDialog("Sauvegarde et mise à jour des paramètres...", "Annuler", 0, 100)
    progress.setWindowTitle("Traitement en cours")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)

    # Création du worker
    worker = _SaveDataWorker(angle_rot_cor, angle_rot_sag, coord_x, coord_y, coord_z,
                             data_repcom, idx, main_dir, min_x, min_y, min_z, num_patient, ref, debug_mode)

    # Connexions
    worker.progress_signal.connect(progress.setValue)
    worker.finished_signal.connect(lambda: progress.setValue(100))
    
    # Lancement
    worker.start()

    # Maintien de l'interface active pendant le thread
    while worker.isRunning():
        app.processEvents()
        if progress.wasCanceled():
            worker.terminate()
            break
            
    worker.wait() # Assure que le thread est bien fermé proprement
    return True

def data_pre_treatment_repcom_2_v2(angle_rot_cor, angle_rot_sag, debug_mode, idx, data_3D, 
                                  main_dir, num_patient, ref, T, origin_coord):
    
    # 1. Dossier de sortie
    output_path = os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Repere_commun', str(ref))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 2. Origine
    print('data shape init', data_3D.shape)
    coord_x, coord_y, coord_z = choose_recons3D_origin_v2(data_3D, origin_coord)
    print('Chosen origin coords:', coord_x, coord_y, coord_z)

    # 3. Padding 

    def pad_pre_post(data, axis, pad_pre=0, pad_post=0):
        pad_pre = int(max(pad_pre, 0))
        pad_post = int(max(pad_post, 0))

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (pad_pre, pad_post)

        return np.pad(data, pad_width, mode='constant')


    # ============================
    # X padding (taille figée)
    # ============================
    sx, sy, sz = data_3D.shape

    if coord_x - 1 > sx - coord_x:
        data_3D = pad_pre_post(
            data_3D,
            axis=0,
            pad_post=2 * coord_x - (sx + 1)
        )

    elif coord_x - 1 < sx - coord_x:
        data_3D = pad_pre_post(
            data_3D,
            axis=0,
            pad_pre=sx - 2 * coord_x + 1
        )

    #print("After x pad", data_3D.shape)


    # ============================
    # Y padding (taille figée)
    # ============================
    _, sy, _ = data_3D.shape   #figer AVANT le pad Y

    if coord_y - 1 > sy - coord_y:
        data_3D = pad_pre_post(
            data_3D,
            axis=1,
            pad_post=2 * coord_y - (sy + 1)
        )

    elif coord_y - 1 < sy - coord_y:
        data_3D = pad_pre_post(
            data_3D,
            axis=1,
            pad_pre=sy - 2 * coord_y + 1
        )

    #print("After y pad", data_3D.shape)


    # ============================
    # Permute [3,2,1]
    # ============================
    data_3D = np.transpose(data_3D, (2, 1, 0))
    print("After permute", data_3D.shape)


    # ============================
    # Z padding (taille figée après permute)
    # ============================
    szp, _, _ = data_3D.shape  # ⚠️ figer AVANT le pad Z

    if coord_z - 1 > szp - coord_z:
        data_3D = pad_pre_post(
            data_3D,
            axis=0,
            pad_post=2 * coord_z - (szp + 1)
        )

    elif coord_z - 1 < szp - coord_z:
        data_3D = pad_pre_post(
            data_3D,
            axis=0,
            pad_pre=szp - 2 * coord_z + 1
        )

    print("After z pad", data_3D.shape)


    # 4. Padding de rotation (Conversion forcée en int)
    pad_x = int(math.ceil((1 / math.cos(2 * math.pi * 30 / 360) - 1) * coord_x))
    pad_y = int(math.ceil((1 / math.cos(2 * math.pi * 5 / 360) - 1) * coord_y))
    pad_z = int(math.ceil((1 / math.cos(2 * math.pi * 30 / 360) - 1) * coord_z))

    #print('pad values', pad_x, pad_y, pad_z)

    # Application sécurisée
    data_3D = np.pad(data_3D, ((pad_z, pad_z), (0, 0), (0, 0)), mode='constant')
    data_3D = np.transpose(data_3D, (2, 1, 0))
    #print('before pad', data_3D.shape)
    data_repcom = np.pad(data_3D, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode='constant')
    #print('after pad', data_repcom.shape)
    
    del data_3D

    # 5. Gestion des rotations (Interface utilisateur)
    output_coord = 'Non'
    # S'assurer qu'une QApplication existe
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication(sys.argv)

    if angle_rot_cor is not None or angle_rot_sag is not None:
        reply = QMessageBox.question(
            None,
            "Repere commun",
            "Souhaitez-vous redefinir les angles ?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

    if reply == QMessageBox.Yes:
        output_coord = 'Oui'
    
    #print('before rot',data_repcom.shape)

    # Rotation Coronale
    if angle_rot_cor is None or output_coord == 'Oui':
        angle_rot_cor = determine_img_angle_rot_reg_v2(data_repcom, 'cor',
                                                       next(Path(os.path.abspath(__file__)).parents[3].rglob("rot_cor.png"), None))
    
    data_repcom = data_rotation_v2(angle_rot_cor, data_repcom, 'cor')
    print('after cor rotation', data_repcom.shape, 'angle:', angle_rot_cor)
    

    # Rotation Sagittale
    if angle_rot_sag is None or output_coord == 'Oui':
        angle_rot_sag = determine_img_angle_rot_reg_v2(data_repcom, 'sag',
                                                       next(Path(os.path.abspath(__file__)).parents[3].rglob("rot_sag.png"), None))
    
    data_repcom = data_rotation_v2(angle_rot_sag, data_repcom, 'sag')
    print('after sag rotation', data_repcom.shape, 'angle:', angle_rot_sag)

    #print("Data shape after initial cropping:", data_repcom.shape)
    print("Cropping. Veuillez patienter...")

    def crop_symmetric_uint8(data):
        data = data.astype(np.uint8)
        
        # 1. Création du masque de l'objet
        def get_clean_mask(data, threshold=10):
            mask = data > threshold
            mask_cleaned = ndimage.binary_erosion(mask, iterations=2)
            mask_cleaned = ndimage.binary_dilation(mask_cleaned, iterations=2)
            return mask_cleaned

        mask_final = get_clean_mask(data)

        # 2. Trouver les limites réelles de l'objet (Bounding Box)
        coords = np.argwhere(mask_final)
        if coords.size == 0:
            return data, 0, 0, 0

        # Limites min/max de l'objet sur chaque axe
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # Tailles actuelles du volume
        sz, sy, sx = data.shape

        def get_symmetric_slice(obj_min, obj_max, full_size):
            """
            Calcule les indices de coupe pour que l'objet soit centré.
            """
            # Le centre de l'objet
            obj_center = (obj_min + obj_max) / 2
            
            # La demi-largeur nécessaire pour contenir l'objet
            half_width = max(obj_center - obj_min, obj_max - obj_center)
            
            # On ajoute une petite marge de sécurité (ex: 20 pixels)
            half_width += 20 
            
            # Nouvelles bornes symétriques par rapport au centre de l'objet
            start = int(max(0, obj_center - half_width))
            end = int(min(full_size, obj_center + half_width))
            
            # Ajustement pour assurer une symétrie parfaite par rapport au centre réel
            # Si on a été limité par un bord (0 ou full_size), on réduit l'autre côté
            dist_to_start = obj_center - start
            dist_to_end = end - obj_center
            actual_half_width = min(dist_to_start, dist_to_end)
            
            final_start = int(obj_center - actual_half_width)
            final_end = int(obj_center + actual_half_width)
            
            return slice(final_start, final_end), final_start

        # Calcul des tranches symétriques pour chaque axe
        slice_z, m_z = get_symmetric_slice(z_min, z_max, sz)
        slice_y, m_y = get_symmetric_slice(y_min, y_max, sy)
        slice_x, m_x = get_symmetric_slice(x_min, x_max, sx)

        return data[slice_z, slice_y, slice_x], m_x, m_y, m_z

    # Utilisation
    data_repcom, min_x, min_y, min_z = crop_symmetric_uint8(data_repcom)



    #print("Cropped shape :", data_repcom.shape)
    print("Sauvegarde. Veuillez patienter...")

    # 7. Sauvegarde
   
   
    save_data_repcom_v2(
        angle_rot_cor, angle_rot_sag, coord_x, coord_y, coord_z, 
        data_repcom, idx, main_dir, min_x, min_y, min_z, num_patient, ref, debug_mode
    )

    valider_recalage(data_repcom)

    return data_repcom


def data_pre_treatment_repcom_v2(angle_rot_cor, angle_rot_sag, debug_mode, idx, 
                                main_dir, num_patient, ref, T, origin_coord):
    """
    Vérifie l'existence et lance le replacement dans le repère commun.
    """
    import os
    from PySide6.QtWidgets import QMessageBox, QApplication
    from PySide6.QtCore import Qt
    import scipy.io as sio
    import numpy as np

    repcom_dir = os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Repere_commun', ref)
    repcom_file = os.path.join(repcom_dir, f'data_repcom_{ref}.mat')
    recon_file = os.path.join(main_dir, 'Pre_traitement_echo_v2', 'Reconstruction_3D', 
                              ref, f'data_3D_{ref}.mat')

    # Si le fichier n'existe pas, on doit traiter (True).
    # Si le fichier existe, on demandera à l'utilisateur.
    should_process = not os.path.exists(repcom_file)

    if os.path.exists(repcom_file):
        diag = QMessageBox()
        diag.setWindowFlags(Qt.WindowStaysOnTopHint)
        diag.setWindowTitle('Fichier existant')
        diag.setText(f"L'examen {ref} a déjà été traité.")
        diag.setInformativeText("Voulez-vous le recalculer [repère commun] ?")
        diag.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        diag.setDefaultButton(QMessageBox.No)
        
        reponse = diag.exec()
        if reponse == QMessageBox.Yes:
            should_process = True
        else:
            should_process = False
            
        diag.deleteLater() # Nettoyage propre
        QApplication.instance().processEvents()

    # Chargement et Traitement
    if should_process:
        if not os.path.exists(recon_file):
            print(f"ERREUR : L'examen {ref} doit être reconstruit en 3D d'abord.")
            return

        print(f"Chargement de la reconstruction 3D : {ref}...")
        try:
            mat_data = sio.loadmat(recon_file)
            data_3d = mat_data.get('data_3D')
            print(f"Data loaded for {ref}, shape: {data_3d.shape}, dtype: {data_3d.dtype}", "min_:" , np.min(data_3d), "max_:", np.max(data_3d))
            del mat_data 

            if data_3d is None:
                print(f"Variable 'data_3D' absente de {recon_file}")
                return
            
            if data_3d.dtype == np.float64:
                data_3d = data_3d.astype(np.float32)

            os.makedirs(repcom_dir, exist_ok=True)

            # Appel du traitement lourd
            data_pre_treatment_repcom_2_v2(
                angle_rot_cor, angle_rot_sag, debug_mode, idx, data_3d, 
                main_dir, num_patient, ref, T, origin_coord
            )
            
            print(f"Succès : Examen {ref} placé dans le repère commun.")

        except Exception as e:
            print(f"Erreur lors du traitement de {ref} : {e}")
        finally:
            QApplication.instance().processEvents()

    return # Fin de la fonction