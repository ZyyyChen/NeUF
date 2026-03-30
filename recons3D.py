from pathlib import Path
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from PySide6.QtWidgets import QMessageBox
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from tools.gui_lay3d import gui_lay3d
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QMessageBox
import os
import sys
from scipy.ndimage import binary_fill_holes
from matplotlib.widgets import Button
import numpy as np
from scipy.spatial import KDTree
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog
import logging
from tools.export_to_3d_dicom import export_to_3d_dicom
from tools.MITKviewer import *
import numpy as np
import matplotlib.gridspec as gridspec
import time



def plot_sag_plan_params(all_num_img_cc, all_rect, coord_pts_sagplan, data_recal, img_cc_ref):
    """
    Affiche la coupe de référence avec ses ROIs et les coupes sélectionnées 
    dans le volume recalé avec leurs points de pivot (Version GridSpec).
    """

    # 1. Préparation des images extraites du volume (normalisation)
    if np.isscalar(all_num_img_cc):
        all_num_img_cc = [all_num_img_cc]
        all_rect = [all_rect]
        coord_pts_sagplan = [coord_pts_sagplan]

    num_imgs = len(all_num_img_cc)
    
    # --- FIGURE 1 : Image sagittale de référence (AGRANDIE via GridSpec) ---
    fig1 = plt.figure(figsize=(20, 10))
    gs1 = gridspec.GridSpec(1, 1) 
    ax1 = fig1.add_subplot(gs1[0])
    
    ax1.imshow(img_cc_ref, cmap='gray')
    ax1.set_title('Image sagittale de référence', fontsize=14, fontweight='bold')
    
    for i in range(num_imgs):
        rect = all_rect[i] # Format [x, y, width, height]
        print(rect)
        rectangle = plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], 
                                  edgecolor='red', facecolor='none', linewidth=2)
        ax1.add_patch(rectangle)
    
    # Positionnement de la fenêtre 1 (Centrée gauche)
    try:
        mngr1 = fig1.canvas.manager
        screen = mngr1.window.screen().geometry()
        mngr1.window.setGeometry(50, 100, 900, 800)
    except:
        pass
    
    # --- FIGURE 2 : Images choisies (AGRANDIES via GridSpec) ---
    fig2 = plt.figure(figsize=(20, 10))
    gs2 = gridspec.GridSpec(1, num_imgs)
    
    axes2 = []
    for i in range(num_imgs):
        ax = fig2.add_subplot(gs2[i])
        
        # Extraction de la tranche
        idx = int(all_num_img_cc[i][0]) if isinstance(all_num_img_cc[i], (list, np.ndarray)) else int(all_num_img_cc[i])
        slice_img = data_recal[:, idx, :].astype(float)
        
        # Normalisation locale
        if slice_img.max() > 0:
            slice_img /= slice_img.max()
            
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f'Image sagittale {i+1} choisie (Index {idx})')
        axes2.append(ax)

    # Positionnement de la fenêtre 2 (Décalée à droite)
    try:
        mngr2 = fig2.canvas.manager
        mngr2.window.setGeometry(1000, 100, 900, 800)
    except:
        pass

    plt.show(block=False) 
    plt.pause(0.1) 

    # 2. Boîte de dialogue (Questdlg)
    msg_box = QMessageBox()
    msg_box.setWindowTitle('Choix des images pour placer des points')
    msg_box.setText('Souhaitez-vous définir de nouvelles images pour placer des points ?')
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    
    reply = msg_box.exec()
    new_img_cc_seqdyn = 'Oui' if reply == QMessageBox.Yes else 'Non'

    # Fermeture des figures
    plt.close(fig1)
    plt.close(fig2)
    
    return new_img_cc_seqdyn


def get_imgs_cc_seqdyn_2(data_recal, img_cc_ref,
                         im_demo_path=next(Path(os.path.abspath(__file__)).parents[3].rglob("select_plan_sag.png"), None)
                         ):
    """
    Sélection interactive avec positionnement automatique des fenêtres côte à côte.
    """
    all_rect = []
    all_num_img_cc = []

    # ==========================================
    # FENÊTRE SÉPARÉE : IMAGE DÉMO
    # ==========================================
    fig_demo = plt.figure("Guide de Référence", figsize=(16, 8))
    try:
        img_demo = mpimg.imread(im_demo_path)
        ax_demo = fig_demo.add_subplot(111)
        ax_demo.imshow(img_demo)
        ax_demo.set_title("DEMO", fontweight='bold', color='forestgreen')
        ax_demo.axis("off")
        
        # --- POSITIONNEMENT GAUCHE ---
        mngr_demo = plt.get_current_fig_manager()
        # setGeometry(x, y, largeur, hauteur)
        mngr_demo.window.setGeometry(50, 100, 1100, 700) 
        fig_demo.show()
    except Exception as e:
        print(f"Note : Image démo non affichée : {e}")

    # ==========================================
    # FENÊTRE PRINCIPALE : INTERFACE
    # ==========================================
    fig = plt.figure("Interface de sélection", figsize=(16, 8))
    
    # --- POSITIONNEMENT DROITE ---
    try:
        mngr_main = fig.canvas.manager
        # On décale la fenêtre de 700 pixels vers la droite pour éviter le chevauchement
        mngr_main.window.setGeometry(1000, 50, 1500, 1000)
    except Exception:
        pass # Si le backend n'est pas Qt, ignore le placement

    gs_main = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[2, 1],
        wspace=0.1, left=0.05, right=0.95, top=0.92, bottom=0.05
    )

    # Axe Référence
    ax_ref = fig.add_subplot(gs_main[0])
    ax_ref.imshow(img_cc_ref, cmap='gray', aspect='equal')
    ax_ref.set_title("Référence", fontweight='bold', fontsize=14, color='royalblue')
    ax_ref.axis("off")

    # Axe GUI interactif
    gs_viewer = gridspec.GridSpecFromSubplotSpec(20, 20, subplot_spec=gs_main[1])
    ax_title_gui = fig.add_subplot(gs_viewer[0, :])
    ax_title_gui.set_title("Séq dyn", fontweight='bold', fontsize=14, color='royalblue', x=0.3)
    ax_title_gui.axis("off")

    fig_v, btn_X, btn_Y, btn_Z, slider, btn_prev, btn_next, switch_mode = \
        gui_lay3d(data_recal, fig=fig, gs=gs_viewer, aspect='auto')

    switch_mode('Y', 1)

    # ==========================================
    # GESTION DU RECTANGLE SELECTOR
    # ==========================================
    current_rect = [0, 0, 0, 0]
    def on_select(eclick, erelease):
        nonlocal current_rect
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2): return
        current_rect = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]

    rs = RectangleSelector(
        ax_ref, on_select, useblit=True, button=[1], 
        interactive=True, props=dict(edgecolor='red', linewidth=2, fill=False)
    )

    # ==========================================
    # 4. BOUCLE D'INTERACTION
    # ==========================================
    plt.show(block=False)

    output_img_cc_2 = 1
    while output_img_cc_2 != 0:
        try:
            val = input("\nNuméro de coupe sagittale (ou '0' pour finir) : ")
            num_img = int(val)
            if num_img == 0 and not all_num_img_cc: 
                print("Veuillez sélectionner au moins une coupe.")
                continue
            
            if num_img == 0: break
            
            all_num_img_cc.append(num_img)

            
            output_img_cc_2 = 0
                
        except ValueError:
            print("Entrée invalide. Entrez un nombre.")

    # Nettoyage et sortie
    plt.close(fig)
    try: plt.close(fig_demo) 
    except: pass

    all_num_img_cc = np.array(all_num_img_cc)
    indices = np.argsort(all_num_img_cc)
    
    return all_num_img_cc[indices], (np.array([[0, 0, 0, 0]]))

def determine_num_img_cc_v2(data_recal, main_dir, ref):
    """
    Détermine les indices d'images et les ROIs pour le plan sagittal de référence.
    """
    app = QApplication.instance()
    main_path = Path(main_dir)
    
    # Chargement de la coupe sagittale de référence
    repos_path = main_path / 'Pre_traitement_echo_v2' / 'Cropping' / ref
    file_sag = repos_path / f"data_repos_{ref}_sag.mat"
    
    if not file_sag.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_sag}")
        
    data_mat = sio.loadmat(str(file_sag))
    img_cc_ref = data_mat['data_sag']

    # Initialisation des variables
    new_img_cc_seqdyn = 'Non'
    all_num_img_cc = None
    all_rect = None

    # 2. Vérification de l'existence des paramètres de reconstruction 3D
    recons_path = main_path / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    file_params = recons_path / f"data_3D_{ref}_sagplan_params.mat"

    if file_params.exists():
        params_mat = sio.loadmat(str(file_params))
        all_num_img_cc = params_mat['all_num_img_cc']
        all_rect = params_mat['all_rect']
        coord_pts_sagplan = params_mat['coord_pts_sagplan']
        
        # Avant d'ouvrir une nouvelle interface, on nettoie ce qui précède
        if app: app.processEvents()

        new_img_cc_seqdyn = plot_sag_plan_params(
            all_num_img_cc, all_rect, coord_pts_sagplan, data_recal, img_cc_ref
        )
        
        # On nettoie APRES l'affichage
        if app: app.processEvents()

    if not file_params.exists() or new_img_cc_seqdyn == 'Oui':
        # Même chose ici avant de lancer la sélection manuelle
        if app: app.processEvents()
        
        all_num_img_cc, all_rect = get_imgs_cc_seqdyn_2(
            data_recal, img_cc_ref
        )
        
        if app: app.processEvents()
        all_num_img_cc = all_num_img_cc[0]
    else:
        all_num_img_cc = all_num_img_cc[0][0]

    return all_num_img_cc, all_rect, img_cc_ref

class PointSelector:
    """Gère la sélection interactive : Clic gauche (+) / Clic droit (-)"""
    def __init__(self, ax):
        self.ax = ax
        self.x_coords = []
        self.y_coords = []
        # Création du marqueur vert ('g+')
        self.line, = self.ax.plot([], [], 'g+', markersize=10, markeredgewidth=1.5)
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1: # Clic gauche : Ajouter
            self.x_coords.append(event.xdata)
            self.y_coords.append(event.ydata)
        elif event.button == 3: # Clic droit : Retirer
            if self.x_coords:
                self.x_coords.pop()
                self.y_coords.pop()
        
        self.line.set_data(self.x_coords, self.y_coords)
        self.ax.figure.canvas.draw()


def determine_transform_points_v2_(img_cc_ref, img_cc_seqdyn, main_dir, ref, mode_debug):
    main_path = Path(main_dir)
    recons_dir = main_path / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    file_points = recons_dir / f"data_3D_{ref}_reconstruction_points.mat"
    
    output_ref_pts = 'Oui'
    coord_pts_img_ref = None
    coord_pts_img_seqdyn = None

    # 1. Vérification si des points existent déjà
    if file_points.exists():
        data_pts = sio.loadmat(str(file_points))
        coord_pts_img_ref = data_pts['coord_pts_img_ref']
        coord_pts_img_seqdyn = data_pts['coord_pts_img_seqdyn']
        
        # Récupération des couleurs sauvegardées
        colors_ref = data_pts.get('colors_ref', ['white']*len(coord_pts_img_ref))
        colors_seq = data_pts.get('colors_seq', ['white']*len(coord_pts_img_seqdyn))

        fig = plt.figure('Vérification des points', figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2]) 
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax1)

        # Affichage Image 1 (Gros plan / Gauche)
        ax1.imshow(img_cc_ref, cmap='gray')
        for i, (pt, col) in enumerate(zip(coord_pts_img_ref, colors_ref)):
            c = col.strip() if isinstance(col, str) else col
            ax1.plot(pt[0], pt[1], '+', color=c, ms=12, mew=3)
            ax1.text(pt[0]+5, pt[1]-5, str(i+1), color=c, fontweight='bold', 
                     bbox=dict(facecolor='black', alpha=0.5, lw=0))
        
        # Affichage Image 2 (Droite)
        ax2.imshow(img_cc_seqdyn, cmap='gray')
        for i, (pt, col) in enumerate(zip(coord_pts_img_seqdyn, colors_seq)):
            c = col.strip() if isinstance(col, str) else col
            ax2.plot(pt[0], pt[1], '+', color=c, ms=12, mew=3)
            ax2.text(pt[0]+5, pt[1]-5, str(i+1), color=c, fontweight='bold', 
                     bbox=dict(facecolor='black', alpha=0.5, lw=0))

        plt.show(block=False); plt.pause(0.1)

        msg_box = QMessageBox()
        msg_box.setText("Utiliser ces points existants ?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = msg_box.exec()
        output_ref_pts = 'Non' if reply == QMessageBox.Yes else 'Oui'
        plt.close('all')

    # 2. Définition des nouveaux points
    if output_ref_pts == 'Oui':
        res1, res2 = data_pre_treatment_recons3D_refpoints_definition_v2(img_cc_ref, img_cc_seqdyn)
        
        coord_pts_img_ref = np.array([[p[0], p[1]] for p in res1])
        coord_pts_img_seqdyn = np.array([[p[0], p[1]] for p in res2])
        colors_ref = [p[2] for p in res1]
        colors_seq = [p[2] for p in res2]

        recons_dir.mkdir(parents=True, exist_ok=True)
        if mode_debug:
            sio.savemat(str(file_points), {
                'coord_pts_img_ref': coord_pts_img_ref,
                'coord_pts_img_seqdyn': coord_pts_img_seqdyn,
                'colors_ref': colors_ref,
                'colors_seq': colors_seq
            })

    return coord_pts_img_ref, coord_pts_img_seqdyn


def data_pre_treatment_recons3D_refpoints_definition_v2(img_cc_ref, img_cc_seqdyn, 
                                                        im_demo_path=next(Path(os.path.abspath(__file__)).parents[3].rglob("select_points.png"), None)
                                                        ):
    colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange', 'red', 'white', 'blue']
    state = {'color': colors[0], 'index': 0}
    pts1, pts2 = [], [] 
    plots1, plots2 = [], [] 
    is_finished = False

    # --- DIMENSIONS ET ESPACEMENT ---
    pal_w, pal_h = 180, 750
    demo_w, demo_h = 900, 600  # Taille de la fenêtre démo
    img_w, img_h = 1300, 850
    gap = 15
    # Largeur totale pour le centrage : Démo + Palette + Images
    total_w = demo_w + gap + pal_w + gap + img_w

    # =========================
    # FENÊTRE DÉMO (NOUVELLE)
    # =========================
    fig_demo = plt.figure('Exemple / Aide', figsize=(4, 4))
    try:
        img_demo = mpimg.imread(im_demo_path)
        ax_demo = fig_demo.add_subplot(111)
        ax_demo.imshow(img_demo)
        ax_demo.set_title("DEMO", fontweight='bold', color='forestgreen')
        ax_demo.axis('off')
    except:
        print("Image démo non trouvée.")

    # =========================
    # FENÊTRE PALETTE
    # =========================
    fig_pal = plt.figure('Palette', figsize=(2, 8))
    ax_t = fig_pal.add_subplot(111); ax_t.axis('off')
    the_table = ax_t.table(cellText=[[""] for _ in colors], colLabels=["COL"], loc='center')
    
    def update_pal():
        for i, color in enumerate(colors):
            cell = the_table[(i + 1, 0)]
            cell.set_facecolor(color)
            cell.set_alpha(1.0 )
            cell.set_linewidth(4 if i == state['index'] else 1)
        fig_pal.canvas.draw_idle()
    
    the_table.scale(1, 4); update_pal()

    # =========================
    # FENÊTRE SAISIE IMAGES
    # =========================
    fig_img = plt.figure('Saisie des Points', figsize=(25, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2]) 
    ax1 = fig_img.add_subplot(gs[0])
    ax2 = fig_img.add_subplot(gs[1], sharey=ax1)
    plt.subplots_adjust(bottom=0.12, top=0.92, wspace=0.1, left=0.05, right=0.95)
    ax1.imshow(img_cc_ref, cmap='gray', aspect='equal'); ax2.imshow(img_cc_seqdyn, cmap='gray', aspect='auto')
    ax1.set_title("Référence", fontweight='bold', color='royalblue'); ax2.set_title("Seq Dyn", fontweight='bold', color="royalblue")
    
    status_ind = fig_img.text(0.5, 0.96, "COULEUR ACTIVE", color='black', fontweight='bold',
                              ha='center', bbox=dict(facecolor=state['color'], alpha=1, boxstyle='round'))

    # --- LOGIQUE INTERACTIVE ---
    def onclick(event):
        if event.inaxes == ax_t: # Clic sur palette
            for (row, col), cell in the_table.get_celld().items():
                if row > 0 and cell.get_window_extent().contains(event.x, event.y):
                    state['index'] = row - 1
                    state['color'] = colors[row-1]
                    status_ind.get_bbox_patch().set_facecolor(state['color'])
                    update_pal(); fig_img.canvas.draw_idle()
            return
        
        if event.inaxes in [ax1, ax2]:
            if event.button == 1: # Clic gauche : Ajout
                ax = event.inaxes
                count = len(pts1)+1 if ax == ax1 else len(pts2)+1
                p, = ax.plot(event.xdata, event.ydata, '+', color=state['color'], ms=14, mew=2)
                t = ax.text(event.xdata+5, event.ydata-5, str(count), color=state['color'], 
                            fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, lw=0))
                if ax == ax1: 
                    pts1.append([event.xdata, event.ydata, state['color']])
                    plots1.append((p, t))
                else: 
                    pts2.append([event.xdata, event.ydata, state['color']])
                    plots2.append((p, t))
            elif event.button == 3: # Clic droit : Annuler
                if event.inaxes == ax1 and pts1:
                    pts1.pop(); [obj.remove() for obj in plots1.pop()]
                elif event.inaxes == ax2 and pts2:
                    pts2.pop(); [obj.remove() for obj in plots2.pop()]
            fig_img.canvas.draw_idle()

    def on_finish(event): nonlocal is_finished; is_finished = True
    btn_ax = plt.axes([0.45, 0.02, 0.1, 0.05])
    btn = Button(btn_ax, 'Valider', color='#CCFFCC')
    btn.on_clicked(on_finish)

    fig_pal.canvas.mpl_connect('button_press_event', onclick)
    fig_img.canvas.mpl_connect('button_press_event', onclick)
    
    # --- POSITIONNEMENT CÔTE À CÔTE ---
    try:
        m_d, m_p, m_i = fig_demo.canvas.manager, fig_pal.canvas.manager, fig_img.canvas.manager
        sc = m_p.window.screen().geometry()
        sx = (sc.width() - total_w) // 2
        sy = (sc.height() - img_h) // 2
        
        # Fenêtre Démo (Gauche)
        m_d.window.setGeometry(sx, sy, demo_w, demo_h)
        # Fenêtre Palette (Milieu)
        m_p.window.setGeometry(sx + demo_w + gap, sy, pal_w, pal_h)
        # Fenêtre Images (Droite)
        m_i.window.setGeometry(sx + demo_w + gap + pal_w + gap, sy, img_w, img_h)
    except: pass

    plt.show(block=False)
    while not is_finished:
        if not (plt.fignum_exists(fig_img.number)): break
        fig_img.canvas.flush_events()
        fig_pal.canvas.flush_events()
        fig_demo.canvas.flush_events()
        time.sleep(0.02)

    plt.close('all')
    return pts1, pts2

def transform_coord(coord_pts_img_sag, coord_pts_img_seqdyn, delta_X_img_sag, delta_X_seqdyn, idx_sag, P, tr_coord):
    """
    Transforme les coordonnées 2D des points d'intérêt en coordonnées 
    physiques et applique le changement de base 3D.
    
    Args:
        coord_pts_img_sag (np.array): Points [x, y] de l'image sagittale.
        coord_pts_img_seqdyn (np.array): Points [x, y] de la séquence dynamique.
        delta_X_img_sag (float): Résolution (mm/px) de l'image sagittale.
        delta_X_seqdyn (float): Résolution (mm/px) de la séquence dynamique.
        idx_sag (int): Index de la coupe sagittale dans le nouveau volume.
        P (np.array): Matrice de passage (3x3).
        tr_coord (np.array): Vecteur de translation (3x1).
    """
    
    # Mise à l'échelle et inversion (X, Y) -> (Ligne, Colonne)
    # coord_pts_img_sag_tr = [coord_pts_img_sag(:,2), coord_pts_img_sag(:,1)];
    scale_factor = delta_X_img_sag / delta_X_seqdyn
    coord_pts_img_sag_scaled = coord_pts_img_sag * scale_factor
    
    # Inversion des colonnes pour passer de (x, y) à (y, x)
    coord_pts_img_sag_tr = np.column_stack((coord_pts_img_sag_scaled[:, 1], 
                                            coord_pts_img_sag_scaled[:, 0]))

    # Transformation des points de la séquence dynamique
    # Inversion (x, y) -> (y, x)
    pts_seq_yx = np.column_stack((coord_pts_img_seqdyn[:, 1], 
                                  coord_pts_img_seqdyn[:, 0]))
    
    # Préparation pour le calcul matriciel 3D : [Y, idx_sag, X]
    num_pts = len(pts_seq_yx)
    # ones(length(...), 1) * idx_sag
    ones_col = np.full((num_pts, 1), idx_sag)
    
    # Assemblage de la matrice 3xN : [Y; idx_sag_repete; X]
    pts_3d = np.vstack((pts_seq_yx[:, 0], 
                        ones_col.flatten(), 
                        pts_seq_yx[:, 1]))
    
    # Application du changement de base : P * (pts_3d + tr_coord - 1)
    # tr_coord doit être de forme (3, 1) pour le broadcasting
    tr_coord = tr_coord.reshape(3, 1)
    pts_3d_transformed = P @ (pts_3d + tr_coord - 1)
    
    # Extraction des composantes 1 et 3 (indices 0 et 2 en Python)
    # coord_pts_img_seqdyn_tr = coord_pts_img_seqdyn([1,3], :)';
    coord_pts_img_seqdyn_tr = pts_3d_transformed[[0, 2], :].T

    return coord_pts_img_sag_tr, coord_pts_img_seqdyn_tr

def determine_transform_v2(coord_pts_img_ref, coord_pts_img_seqdyn, data_recal,
                                        debug_mode, idx, main_dir):
    """
    MATLAB-like version of determine_transform_v2 using least_squares.
    """
    # --- 1. Bounds (same as MATLAB) ---
    sz3 = data_recal.shape[2]
    b_inf = np.array([-30, 200, -0.08, -2, -80, (-np.pi / 2) / sz3, -np.pi / 3], dtype=np.float64)
    b_sup = np.array([80, 800, 0.08, 2, 30, (np.pi / 2) / sz3, np.pi / 3], dtype=np.float64)

    # --- Initial guess ---
    x0 = np.array([0, 350, 0, 0, 0, 0, 0], dtype=np.float64)
    print(f"Initial x0: {x0}")

    # --- Extract coordinates ---
    r = coord_pts_img_seqdyn[:, 0].astype(np.float64)
    t = coord_pts_img_seqdyn[:, 1].astype(np.float64)
    ref_x = coord_pts_img_ref[:, 0].astype(np.float64)
    ref_y = coord_pts_img_ref[:, 1].astype(np.float64)

    # --- Define residual function (column-major style) ---
    def fun(x):
        # x = [Cx, Ct, vx, vt, delta, omega, theta]
        res_x = x[0] + x[2]*t + (r + x[4]) * np.cos(t*x[5] + x[6]) - ref_x
        res_y = x[1] + x[3]*t + (r + x[4]) * np.sin(t*x[5] + x[6]) - ref_y
        # Scale residuals to balance X and Y
        res = np.empty(res_x.size*2, dtype=np.float64)
        res[0::2] = res_x
        res[1::2] = res_y
        return res

    # --- Define finite-difference step to mimic MATLAB ---
    diff_step = np.full_like(x0, 1e-6, dtype=np.float64)

    # --- Optimization using 'trf' like MATLAB ---
    res = least_squares(
        fun,
        x0,
        bounds=(b_inf, b_sup),
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=2000,
        diff_step=diff_step,
        loss='linear'
    )

    X_transform = res.x
    print(f"Found parameters: {X_transform}")

    # --- Save parameters ---
    if debug_mode:
        main_path = Path(main_dir)
        ref_file_path = main_path / 'Ref_Files' / 'ReconstructionParameters_v2.mat'
        if ref_file_path.exists():
            mat_data = sio.loadmat(str(ref_file_path))
            params_patients = mat_data['params_patients_echo_v2']
        else:
            #params_patients = np.zeros((idx + 1, 7), dtype=object)
            raise FileNotFoundError()

        for i in range(7):
            params_patients[idx, i] = X_transform[i]

        sio.savemat(str(main_path / 'Ref_Files' / 'ReconstructionParameters_v2.mat'), 
                    {'params_patients_echo_v2': params_patients})
        print("Parameters saved in .mat file.")

    return X_transform

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing

def get_data_int_mask(data_recal, pts_aug, xp, zp, closing_size=3):
    """
    Version plus vectorisée + ajout d'une fermeture morphologique (closing)
    
    Parameters:
    -----------
    data_recal : ndarray
        Utilisé pour récupérer la dimension Y
    pts_aug : int
        Nombre de points d'interpolation entre deux points consécutifs
    xp, zp : ndarray (n_points × n_lines)
        Coordonnées X et Z
    closing_size : int, optional
        Taille du kernel de fermeture morphologique (disque de diamètre closing_size)
        
    Returns:
    --------
    data_int_mask : ndarray (uint8)
        Masque binaire 3D après remplissage + closing
    """
    app= QApplication.instance() or QApplication(sys.argv)
    app.processEvents()
    # Dimensions
    x_max = int(np.ceil(xp.max())) + 1
    z_max = int(np.ceil(zp.max())) + 1
    ny = data_recal.shape[1]

    mask = np.zeros((x_max, ny, z_max), dtype=bool)

    # ────────────────────────────────────────────────
    # Placement des points originaux (vectorisé)
    # ────────────────────────────────────────────────
    x_orig = np.round(xp).astype(int)
    z_orig = np.round(zp).astype(int)

    valid = (x_orig >= 0) & (x_orig < x_max) & (z_orig >= 0) & (z_orig < z_max)
    mask[x_orig[valid], :, z_orig[valid]] = True

    # ────────────────────────────────────────────────
    # Interpolation linéaire vectorisée sur toutes les segments
    # ────────────────────────────────────────────────
    def interpolate_line(coords_start, coords_end, n_interp):
        # coords_start, coords_end : shape (n_segments,)
        # → retourne shape (n_segments * n_interp,)
        t = np.linspace(0, 1, n_interp + 1)[:-1]  # exclut le dernier point (déjà dans end)
        t = t[:, None]                             # (n_interp, 1)
        
        start = coords_start[None, :]              # (1, n_segments)
        end   = coords_end[None, :]                # (1, n_segments)
        
        interp = (1 - t) * start + t * end        # (n_interp, n_segments)
        return interp.ravel()                      # (n_interp * n_segments,)

    # Pour la ligne du haut (max)
    xp_max_start = xp[-1, :-1]
    xp_max_end   = xp[-1, 1:]
    zp_max_start = zp[-1, :-1]
    zp_max_end   = zp[-1, 1:]

    xp_max_interp = interpolate_line(xp_max_start, xp_max_end, pts_aug)
    zp_max_interp = interpolate_line(zp_max_start, zp_max_end, pts_aug)

    # Pour la ligne du bas (min)
    xp_min_start = xp[0, :-1]
    xp_min_end   = xp[0, 1:]
    zp_min_start = zp[0, :-1]
    zp_min_end   = zp[0, 1:]

    xp_min_interp = interpolate_line(xp_min_start, xp_min_end, pts_aug)
    zp_min_interp = interpolate_line(zp_min_start, zp_min_end, pts_aug)

    # Placement des points interpolés (haut + bas)
    for x_interp, z_interp in [(xp_max_interp, zp_max_interp),
                               (xp_min_interp, zp_min_interp)]:
        x_r = np.round(x_interp).astype(int)
        z_r = np.round(z_interp).astype(int)
        valid = (x_r >= 0) & (x_r < x_max) & (z_r >= 0) & (z_r < z_max)
        mask[x_r[valid], :, z_r[valid]] = True
        app.processEvents()

    # ────────────────────────────────────────────────
    # Remplissage des trous + fermeture morphologique
    #    (slice par slice selon Y)
    # ────────────────────────────────────────────────
    structure = np.ones((closing_size, closing_size), dtype=bool)  # disque approx

    for iy in range(ny):
        slice_2d = mask[:, iy, :]

        # Remplissage des trous
        filled = binary_fill_holes(slice_2d)

        # Fermeture morphologique (dilatation puis érosion)
        closed = binary_closing(filled, structure=structure)

        mask[:, iy, :] = closed
        app.processEvents()

    return mask.astype(np.uint8)


def data_transform_v2(data_recal, X_transform):
    """
    Applique la transformation géométrique curviligne au volume de données.
    Calcule les nouvelles grilles de coordonnées (x, y) et les offsets.
    """
    # X_transform : [Cx, Ct, vx, vt, delta, omega, theta]
    Cx, Ct, vx, vt, delta, omega, theta = X_transform
    
    n_rows = data_recal.shape[0] # Correspond à (1:size_data(1))
    n_cols = data_recal.shape[2] # Correspond à (1:size_data(3))
    
    # Création des vecteurs de base (1-based pour correspondre à MATLAB)
    # r correspond à la profondeur (axe vertical de l'image)
    # t correspond au balayage temporel ou spatial (axe horizontal)
    r = np.arange(1, n_rows + 1).reshape(-1, 1) # Vecteur colonne (n_rows, 1)
    t = np.arange(1, n_cols + 1).reshape(1, -1) # Vecteur ligne (1, n_cols)
    
    # Application du modèle trigonométrique
    
    x_moving = Cx + vx * t + (r + delta) * np.cos(t * omega + theta)
    
    y_moving = Ct + vt * t + (r + delta) * np.sin(t * omega + theta)
    
    #  Calcul des offsets (pour éviter les indices négatifs lors de l'interpolation)
    offset_x = np.abs(np.minimum(np.min(x_moving), 0)) + 1
    offset_y = np.abs(np.minimum(np.min(y_moving), 0)) + 1
    
    # 4. Ajustement final des grilles
    x_moving += offset_x
    y_moving += offset_y
    
    return x_moving, y_moving, offset_x, offset_y


import numpy as np
from scipy.spatial import KDTree

def data_interpolation_v2_25D_3(data_recal, nb_pts, multi_img, xp, zp):
    """
    Interpolation volumique 2.5D avec recherche de plus proches voisins
    (traduction corrigée et fidèle de la fonction MATLAB data_interpolation_v2_25D_3)
    """
    # ───────────────────────────────────────────────
    # Progress dialog
    # ───────────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)
    app.processEvents()
    wb = QProgressDialog(
        "Calcul du masque d'interpolation (1/3)",
        None, 0, 100, None
    )
    wb.setWindowTitle("Reconstruction 3D")
    wb.setWindowModality(Qt.WindowModality.ApplicationModal)
    wb.setCancelButton(None)
    wb.setMinimumDuration(0)
    wb.show()
    wb.setValue(5)
    app.processEvents()

    print("Calcul du masque d'interpolation (1/3)")
    wb.setLabelText("Calcul du masque d'interpolation (1/3)")
    wb.setValue(20)
    # Création du masque d'interpolation
    app.processEvents()
    data_int_mask_full = get_data_int_mask(data_recal, pts_aug=9, xp=xp, zp=zp)
    app.processEvents()
    data_3D_mask = data_int_mask_full.astype(np.uint8)

    # Coupe centrale pour calculs
    data_int_mask = data_int_mask_full[:, 0, :]

    # Coordonnées transformées
    coord_transform = coord_after_transform(data_recal[:, 0, :], xp, zp)
    coord_transform = coord_transform[:, [0, 2]]  # colonnes 1 et 3 (0-based)

    # Coordonnées à remplir
    coord_mask, idx_mask = coord_to_fill(data_int_mask)
    coord_mask = coord_mask[:, [0, 2]]

    n_voxels = len(coord_mask)
    n_height = data_recal.shape[0]
    n_long = data_recal.shape[2]

    # Recherche KNN par coupe longitudinale
    Idx = np.zeros((n_voxels, n_long, nb_pts), dtype=int)
    Dist = np.zeros((n_voxels, n_long, nb_pts), dtype=float)
    app.processEvents()
    print("Recherche des plus proches voisins (2/3)")
    wb.setLabelText("Recherche des plus proches voisins (2/3)")
    wb.setValue(20)
    app.processEvents()
    for i in range(n_long):
        start = i * n_height
        end = start + n_height
        points_i = coord_transform[start:end, :]
        tree = KDTree(points_i)
        dist_i, idx_i = tree.query(coord_mask, k=nb_pts)
        Idx[:, i, :] = idx_i
        Dist[:, i, :] = dist_i 
        wb.setValue(20 + int(30 * (i + 1) / n_long))
        app.processEvents()

    # 5. Sélection des coupes les plus proches + voisines
    num_img_d_min = np.argmin(Dist[:, :, 0], axis=1)  # 0-based

    if multi_img == 1:
        n_channels = 4
    else:
        n_channels = 2

    I_complet = np.zeros((n_voxels, nb_pts, n_channels), dtype=int)
    d_complet = np.full((n_voxels, nb_pts, n_channels), np.inf)

    for i in range(n_voxels):
        mid = num_img_d_min[i]

        # Channel 0: coupe la plus proche
        if 0 <= mid < n_long:
            I_complet[i, :, 0] = mid * n_height + Idx[i, mid, :]
            d_complet[i, :, 0] = Dist[i, mid, :]

        # Détermination de la direction prioritaire
        bool_right = False
        if mid > 0 and mid < n_long - 1:
            bool_right = Dist[i, mid + 1, 0] < Dist[i, mid - 1, 0]

        if bool_right or mid == 0:
            # Channel 1: voisin droit
            if mid + 1 < n_long:
                I_complet[i, :, 1] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                d_complet[i, :, 1] = Dist[i, mid + 1, :]

            if multi_img == 1:
                if mid == 0:
                    if mid + 2 < n_long:
                        I_complet[i, :, 2] = (mid + 2) * n_height + Idx[i, mid + 2, :]
                        d_complet[i, :, 2] = Dist[i, mid + 2, :]
                    # Channel 3 reste inf
                elif mid == n_long - 2:
                    if mid - 1 >= 0:
                        I_complet[i, :, 2] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                        d_complet[i, :, 2] = Dist[i, mid - 1, :]
                    # Channel 3 inf
                else:
                    if mid - 1 >= 0:
                        I_complet[i, :, 2] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                        d_complet[i, :, 2] = Dist[i, mid - 1, :]
                    if mid + 2 < n_long:
                        I_complet[i, :, 3] = (mid + 2) * n_height + Idx[i, mid + 2, :]
                        d_complet[i, :, 3] = Dist[i, mid + 2, :]

        else:
            # Channel 1: voisin gauche
            if mid - 1 >= 0:
                I_complet[i, :, 1] = (mid - 1) * n_height + Idx[i, mid - 1, :]
                d_complet[i, :, 1] = Dist[i, mid - 1, :]

            if multi_img == 1:
                if mid == 1:
                    if mid + 1 < n_long:
                        I_complet[i, :, 2] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                        d_complet[i, :, 2] = Dist[i, mid + 1, :]
                    # 3 inf
                elif mid == n_long - 1:
                    if mid - 2 >= 0:
                        I_complet[i, :, 2] = (mid - 2) * n_height + Idx[i, mid - 2, :]
                        d_complet[i, :, 2] = Dist[i, mid - 2, :]
                    # 3 inf
                else:
                    if mid + 1 < n_long:
                        I_complet[i, :, 2] = (mid + 1) * n_height + Idx[i, mid + 1, :]
                        d_complet[i, :, 2] = Dist[i, mid + 1, :]
                    if mid - 2 >= 0:
                        I_complet[i, :, 3] = (mid - 2) * n_height + Idx[i, mid - 2, :]
                        d_complet[i, :, 3] = Dist[i, mid - 2, :]
        app.processEvents() 

    # Sécurité (debug)
    max_idx = n_height * n_long - 1
    I_complet = np.clip(I_complet, 0, max_idx)

    # 6. Reshape comme dans MATLAB
    I_complet_1 = I_complet[:, :, 0:2].reshape(n_voxels, nb_pts * 2)
    d_complet_1 = d_complet[:, :, 0:2].reshape(n_voxels, nb_pts * 2)

    if multi_img == 1:
        if nb_pts == 3:
            n_pts_2 = 1
        elif nb_pts == 5:
            n_pts_2 = 3
        else:
            raise ValueError("nb_pts must be 3 or 5 for multi_img=1")
        I_complet_2 = I_complet[:, 0:n_pts_2, 2:4].reshape(n_voxels, n_pts_2 * 2)
        d_complet_2 = d_complet[:, 0:n_pts_2, 2:4].reshape(n_voxels, n_pts_2 * 2)

    # 7. Interpolation
    x_max = int(np.ceil(np.max(xp))) + 1
    z_max = int(np.ceil(np.max(zp))) + 1
    data_int = np.zeros((x_max, data_recal.shape[1], z_max), dtype=float)

    print("Reconstruction et interpolation du volume (3/3)")
    wb.setLabelText("Reconstruction et interpolation du volume (3/3)")

    for j in range(2, data_recal.shape[1] - 2):
        accum = np.zeros(n_voxels)

        data_recal_j = data_recal[:, j, :]

        # Préparation des slices voisines et distances augmentées
        if nb_pts in [3, 5]:
            data_recal_m1 = data_recal[:, j-1, :]
            data_recal_p1 = data_recal[:, j+1, :]
            d_complet_1_1 = np.sqrt(d_complet_1 ** 2 + 1)

        if nb_pts == 5:
            data_recal_m2 = data_recal[:, j-2, :]
            data_recal_p2 = data_recal[:, j+2, :]
            d_complet_1_2 = np.sqrt(d_complet_1 ** 2 + 4)

            if multi_img == 1:
                d_complet_2_1 = np.sqrt(d_complet_2 ** 2 + 1)

        # Accumulation principale (channels 1 et 2)
        for k in range(I_complet_1.shape[1]):
            idx_k = I_complet_1[:, k]
            w = 1 / d_complet_1[:, k]
            vals = data_recal_j.ravel()[idx_k]
            accum += w * vals

            if nb_pts == 3:
                w1 = 1 / d_complet_1_1[:, k]
                vals_m1 = data_recal_m1.ravel()[idx_k]
                vals_p1 = data_recal_p1.ravel()[idx_k]
                accum += w1 * (vals_m1 + vals_p1)
            elif nb_pts == 5:
                w1 = 1 / d_complet_1_1[:, k]
                vals_m1 = data_recal_m1.ravel()[idx_k]
                vals_p1 = data_recal_p1.ravel()[idx_k]
                accum += w1 * (vals_m1 + vals_p1)

                w2 = 1 / d_complet_1_2[:, k]
                vals_m2 = data_recal_m2.ravel()[idx_k]
                vals_p2 = data_recal_p2.ravel()[idx_k]
                accum += w2 * (vals_m2 + vals_p2)

        # Accumulation additionnelle (channels 3 et 4 si multi_img)
        if multi_img == 1:
            for k in range(I_complet_2.shape[1]):
                idx_k = I_complet_2[:, k]
                w = 1 / d_complet_2[:, k]
                vals = data_recal_j.ravel()[idx_k]
                accum += w * vals

                if nb_pts == 5:
                    w1 = 1 / d_complet_2_1[:, k]
                    vals_m1 = data_recal_m1.ravel()[idx_k]
                    vals_p1 = data_recal_p1.ravel()[idx_k]
                    accum += w1 * (vals_m1 + vals_p1)

        # Normalisation
        sum_w = np.sum(1 / d_complet_1, axis=1)

        if nb_pts in [3, 5]:
            sum_w += 2 * np.sum(1 / d_complet_1_1, axis=1)

        if nb_pts == 5:
            sum_w += 2 * np.sum(1 / d_complet_1_2, axis=1)

        if multi_img == 1:
            sum_w += np.sum(1 / d_complet_2, axis=1)
            if nb_pts == 5:
                sum_w += 2 * np.sum(1 / d_complet_2_1, axis=1)

        sum_w[sum_w == 0] = 1  # Éviter div/0
        accum /= sum_w

        data_int_temp = np.zeros((x_max, z_max))
        data_int_temp.ravel()[idx_mask] = accum 

        data_int[:, j, :] = data_int_temp
        if j % 10 == 0:
            prog = 50 + int(45 * (j / data_recal.shape[1]))
            wb.setValue(min(prog, 98))
            app.processEvents()

    data_int = data_int.astype(np.uint8)
    wb.setValue(min(prog, 100))
    app.processEvents()

    return data_int, data_3D_mask

def coord_after_transform(data_recal, xp, zp):
    """
    Python equivalent of MATLAB coord_after_transform
    data_recal: shape (Nx, Nz)
    xp: shape (Nx, Nz)
    zp: shape (Nx, Nz)
    """

    Nx, Nz = data_recal.shape

    xp = xp.reshape(Nx, 1, Nz)
    zp = zp.reshape(Nx, 1, Nz)

    # Allocate
    X = np.zeros((Nx, 1, Nz))
    Y = np.zeros((Nx, 1, Nz))
    Z = np.zeros((Nx, 1, Nz))

    X[:, 0, :] = xp[:, 0, :]
    Y[:, 0, :] = 1        
    Z[:, 0, :] = zp[:, 0, :]

    # Vectorize like MATLAB X(:), Y(:), Z(:)
    coord_transform = np.column_stack((
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ))

    return coord_transform

def coord_to_fill(data_int_mask):

    idx_mask = np.flatnonzero(
        data_int_mask.ravel() > 0
    )

    X, Z = data_int_mask.shape

    x_coords = np.repeat(
        np.arange(1, X + 1)[:, None],
        Z,
        axis=1
    )

    z_coords = np.repeat(
        np.arange(1, Z + 1)[None, :],
        X,
        axis=0
    )

    # Y fixed to 1 (single slice)
    y_coords = np.ones_like(x_coords)

    # Vectorize in Fortran order
    x_flat = x_coords.ravel()[idx_mask]
    y_flat = y_coords.ravel()[idx_mask]
    z_flat = z_coords.ravel()[idx_mask]

    coord_mask = np.column_stack((x_flat, y_flat, z_flat))

    return coord_mask, idx_mask


def save_3Drecons_data(data_3D, data_3D_mask, delta_X_seqdyn, main_dir, num_patient, ref):
    """
    Sauvegarde optimisée du volume 3D avec gestion de la réactivité GUI.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 1. Gestion des chemins
    output_path = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Enregistrement des données : Patient {ref}...")
    app.processEvents()

    try:
        
        # --- A. Sauvegarde .mat ---
        mat_file = output_path / f"data_3D_{ref}.mat"
        
        app.processEvents()
        
        save_dict = {
            'data_3D': data_3D,
            'data_3D_mask': data_3D_mask, # Ajout du masque si nécessaire pour le post-processing
            'metadata': {
                'delta_X': delta_X_seqdyn,
                'patient': num_patient,
                'ref': ref
            }
        }
        
        sio.savemat(str(mat_file), save_dict, do_compression=True)
        print(f"-> Succès : {mat_file.name} enregistré.")
        app.processEvents()

        # --- B. Sauvegarde dicom 3D ---

        # Normalisation et conversion en uint8 (0-255)
        # Note : On utilise 255 car .mhd spécifie MET_UCHAR (unsigned char)
        max_val = np.max(data_3D)
        if max_val > 0:
            # On crée une seule version de données pour le DICOM et le MHD
            data_uint8 = (data_3D / max_val * 255).astype(np.uint8)
        else:
            data_uint8 = data_3D.astype(np.uint8)


        # Rafraîchissement de l'interface GUI
        app.processEvents()

        # Export Dicom 3D
        base_name = f"data_3D_{ref}"
        print(f"-> Succès : Fichiers .mhd/.raw enregistrés dans {output_path}")

        # Préparation des chemins
        output_dicom_file = os.path.join(output_path, f"{base_name}.dcm")

        # Export vers le DICOM 3D Unique (Utilise les données 0-255)
        export_to_3d_dicom(
            data=data_uint8, 
            output_filepath=output_dicom_file, 
            patient_name=str(num_patient), 
            patient_id=str(ref)
        )
        app.processEvents()

        # lancer MITK-style viewer
        valider_recalage(data_3D)

    except Exception as e:
        print(f"ERREUR lors de la sauvegarde : {str(e)}")

    finally:
        # Toujours s'assurer que l'UI est rafraîchie à la fin du processus lourd
        app.processEvents()

def data_recons3D_save_sagplan_parameters(all_num_img_cc, all_rect, coord_pts_sagplan, 
                                         idx_sag, img_cc_seqdyn, 
                                          main_dir, P, ref, tr_coord):
    """
    Sauvegarde les paramètres du plan sagittal calculés dans un fichier .mat.
    """
    # Si nous ne sommes pas en mode démo, on procède à l'enregistrement
    
    # Construction du chemin avec Pathlib
    output_dir = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    
    # Création du dossier si nécessaire
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"data_3D_{ref}_sagplan_params.mat"
    
    # Création du dictionnaire de données pour le .mat
    # On s'assure que les objets numpy sont bien passés
    data_to_save = {
        'all_num_img_cc': all_num_img_cc,
        'all_rect': all_rect,
        'coord_pts_sagplan': coord_pts_sagplan,
        'idx_sag': idx_sag,
        'img_cc_seqdyn': img_cc_seqdyn,
        'P': P,
        'tr_coord': tr_coord
    }
    
    try:
        # Sauvegarde au format .mat
        sio.savemat(str(file_path), data_to_save)
        print(f"Paramètres sauvegardés avec succès : {file_path.name}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")


# Configuration du logger pour le suivi des étapes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_pre_treatment_recons3D_2_v2(delta_X_cc, delta_X_seqdyn, debug_mode, main_dir, 
                                     data_recal, idx, num_patient, ref):
    """
    Reconstruction 3D.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Dossier de sortie
    path_recons = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    path_recons.mkdir(parents=True, exist_ok=True)

    app.processEvents()

    # Définition du plan et transformation
    output_sag_plan = 0
    try:
        # Simulation de la boucle de sélection (adapté de votre logique)
        while not output_sag_plan:
            
            all_num_img_cc, all_rect, img_cc_ref = determine_num_img_cc_v2(
                data_recal, str(main_dir), ref
            )
            
            output_sag_plan = 1
            idx_sag = all_num_img_cc
            coord_pts_sagplan = np.array([[1, idx_sag, 1], [10, idx_sag, 10], [20, idx_sag, 20]])
            
            # Normalisation sécurisée
            img_slice = data_recal[:, idx_sag, :].astype(float)
            max_val = np.max(img_slice)
            img_cc_seqdyn = img_slice / max_val if max_val > 0 else img_slice
            P, tr_coord = np.eye(3), np.ones((3, 1))

        if debug_mode:
            # Sauvegarde des paramètres du plan
            data_recons3D_save_sagplan_parameters(
                all_num_img_cc, all_rect, coord_pts_sagplan, idx_sag, 
                img_cc_seqdyn, str(main_dir), P, ref, tr_coord
            )

        # Calculs géométriques
        coord_pts_img_ref, coord_pts_img_seqdyn = determine_transform_points_v2_(
            img_cc_ref, img_cc_seqdyn, str(main_dir), ref, debug_mode
        )
        
        coord_pts_img_ref_tr, coord_pts_img_seqdyn_tr = transform_coord(
            coord_pts_img_ref, coord_pts_img_seqdyn, delta_X_cc, delta_X_seqdyn, idx_sag, P, tr_coord
        )
        
        X_transform = determine_transform_v2(
            coord_pts_img_ref_tr, coord_pts_img_seqdyn_tr, data_recal, debug_mode, idx, str(main_dir)
        )

        # Transformation et Interpolation (Phase critique pour la RAM)
        logger.info("Transformation du volume...")
        xp, zp, offset_x, offset_y = data_transform_v2(data_recal, X_transform)

        # --- CALCUL DE L'ERREUR DE RECALAGE ---
        r = coord_pts_img_seqdyn_tr[:, 0]
        t = coord_pts_img_seqdyn_tr[:, 1]

        # Calcul des coordonnées recalées
        coord_x_recal = X_transform[0] + X_transform[2]*t + (r + X_transform[4]) * np.cos(t * X_transform[5] + X_transform[6])
        coord_y_recal = X_transform[1] + X_transform[3]*t + (r + X_transform[4]) * np.sin(t * X_transform[5] + X_transform[6])

        # Normalisation des coordonnées (gestion des offsets négatifs)
        offset_x = np.abs(np.minimum(np.min(coord_x_recal), 0)) + 1
        offset_y = np.abs(np.minimum(np.min(coord_y_recal), 0)) + 1
        coord_x_recal += offset_x
        coord_y_recal += offset_y

        # Calcul des erreurs (Différence entre points de réf et points recalés)
        # Note : On garde la logique x/y de votre script MATLAB
        erreur_x = np.abs(coord_x_recal - coord_pts_img_ref_tr[:, 0])
        erreur_y = np.abs(coord_y_recal - coord_pts_img_ref_tr[:, 1])
        
        distances = np.sqrt(erreur_x**2 + erreur_y**2)
        mean_error = np.mean(distances)
        
        # Mean Absolute Deviation (MAD)
        mean_absolute_deviation = np.mean(np.abs(distances - np.mean(distances)))

        print(f"--- Statistiques de recalage ({ref}) ---")
        print(f"Erreur moyenne en pixel : {mean_error:.4f}")
        print(f"Écart moyen erreur en pixel : {mean_absolute_deviation:.4f}")
        # ---------------------------------------
        
        app.processEvents() # Évite le "Ne répond pas"
        
        logger.info("Interpolation 3D en cours...")
        # On passe en float32 pour économiser 50% de RAM par rapport au float64 par défaut
        data_3D, data_3D_mask = data_interpolation_v2_25D_3(
            data_recal.astype(np.float32), 3, 0, xp, zp
        )
        
        # Sauvegarde finale
        save_3Drecons_data(data_3D, data_3D_mask, delta_X_seqdyn, str(main_dir), num_patient, ref)

        if debug_mode:
            # --- Visualisation des points de recalage ---
            
            plt.figure("Vérification reconstruction", figsize=(16, 8), dpi=200)
            tranche = np.squeeze(data_3D[:, idx_sag, :])
            plt.imshow(tranche, cmap='gray')
            
            # Application des offsets (assurez-vous que offset_x/y sont définis)
            coord_pts_img_ref_tr[:, 0] += offset_x 
            coord_pts_img_ref_tr[:, 1] += offset_y
            
            # Affichage des points
            plt.scatter(coord_pts_img_ref_tr[:, 1], coord_pts_img_ref_tr[:, 0], 
                        edgecolors=[0, 1, 0], facecolors='none', s=50, label='Ref TR')
            plt.scatter(coord_y_recal, coord_x_recal, # L'offset est déjà inclus dans ton calcul précédent
                        edgecolors=[1, 0.7, 0.3], facecolors='none', s=50, label='Recal')
            
            plt.legend()

            # --- MISE AU PREMIER PLAN ---
            manager = plt.get_current_fig_manager()
            try:
                manager.window.activateWindow()
                manager.window.raise_()
            except Exception:
                pass

            # --- SAUVEGARDE ---
            output_path = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
            output_path.mkdir(parents=True, exist_ok=True) 
            
            save_file = output_path / "Verification_reconstruction.png"
            plt.savefig(save_file)
            print(f"Figure sauvegardée sous : {save_file}")

            # Affichage
            plt.show(block=True)
            # plt.close(fig) # Décommente cette ligne si tu veux que la fenêtre se ferme seule

        # NETTOYAGE MÉMOIRE CRITIQUE
        del data_3D, data_3D_mask, xp, zp
        import gc
        gc.collect() 
        
        logger.info(f"Reconstruction terminée pour {ref}")
        return True

    except Exception as e:
        logger.error(f"Erreur durant la reconstruction de {ref} : {e}")
        return False

def data_pre_treatment_recons3D_v2(d, delta_X_cc, delta_X_seqdyn, debug_mode, idx, main_dir, num_patient, ref):
    """
    Wrapper de l'étape 3 : Gère les fichiers et le flux d'exécution.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False) # INDISPENSABLE pour ne pas quitter entre les étapes
    
    path_recons = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Reconstruction_3D' / ref
    file_3d = path_recons / f"data_3D_{ref}.mat"
    file_recal = Path(main_dir) / 'Pre_traitement_echo_v2' / 'Recalage' / ref / f"data_recal_{ref}_d_{d}.mat"

    should_reconstruct = not file_3d.exists()

    if file_3d.exists():
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Fichier existant")
        msg_box.setText(f"La reconstruction 3D pour {ref} existe déjà.")
        msg_box.setInformativeText("Voulez-vous la recalculer ?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        if msg_box.exec() == QMessageBox.Yes:
            should_reconstruct = True
        msg_box.deleteLater()

    if should_reconstruct:
        if not file_recal.exists():
            logger.error(f"Erreur : Le fichier de recalage est introuvable ({file_recal.name})")
            return

        logger.info(f"Chargement des données recalées...")
        try:
            # Chargement 
            mat_contents = sio.loadmat(str(file_recal))
            data_recal = mat_contents['data_recal']
            del mat_contents # Libère le dictionnaire temporaire
            
            # Appel du moteur de reconstruction
            data_pre_treatment_recons3D_2_v2(
                delta_X_cc, delta_X_seqdyn, debug_mode, str(main_dir), 
                data_recal, idx, num_patient, ref
            )
        except Exception as e:
            logger.error(f"Erreur fatale : {e}")
        finally:
            app.processEvents()