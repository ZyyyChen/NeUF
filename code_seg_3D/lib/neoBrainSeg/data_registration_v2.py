import numpy as np
from scipy.ndimage import affine_transform, zoom, gaussian_filter, gaussian_filter1d
from scipy import ndimage
from scipy.optimize import minimize
from PySide6.QtWidgets import QApplication, QProgressDialog
from PySide6.QtCore import Qt
#from lib.neoBrainSeg.denoise_data_2D import denoise_volume_2d_parallel
from skimage.morphology import convex_hull_image


def data_registration_v2(d, data_repos):
    if data_repos is None: 
        return None, {}

    # Récupération de l'application sans en créer une nouvelle
    app = QApplication.instance()
    height, width, num_images = data_repos.shape
    img_ref_idx = int(num_images // 2)

    # --- ÉTAPE 0 : RECONSTRUCTION DU BORD DROIT ---
    std_map = np.std(data_repos, axis=2)
    mask = np.where(std_map > 1.5, 1.0, 0.0).astype(np.uint8)
    
    label_im, nb_labels = ndimage.label(mask)
    if nb_labels > 0:
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        biggest_label = np.argmax(sizes[1:]) + 1
        mask = (label_im == biggest_label)
    
    # Enveloppe convexe
    mask_repaired = convex_hull_image(mask).astype(np.float32)
    #h_m, w_m = mask_repaired.shape

    # AJOUT D'UNE MARGE DE SÉCURITÉ (Dilation)
    # On élargit le masque de 15 pixels pour ne pas couper les bords du cerveau
    mask_repaired = ndimage.binary_dilation(mask_repaired, iterations=30).astype(np.float32)
    
    # Application du masque sur les données originales
    data_clean = data_repos * mask_repaired[:, :, np.newaxis]

    # --- 1. DÉBRUITAGE ---
    #print("Débruitage DT-CWT en cours...")
    #data_clean = denoise_volume_2d_parallel(data_clean, T=10) * mask_repaired[:, :, np.newaxis]
    #from tools.MITKviewer import valider_recalage
    #valider_recalage(data_clean)
    
    data_rz = zoom(data_clean, (d, d, 1), order=1).astype(np.float32)
    h_rz, w_rz = data_rz.shape[0], data_rz.shape[1]
    center_rz = np.array([h_rz / 2.0, w_rz / 2.0])
    tform_params = np.zeros((num_images, 3)) 

    def get_transform_powell(fixed, moving):
        current_p = np.array([0.0, 0.0, 0.0])
        sigmas = [4.0, 2.0, 0.0]
        for sig in sigmas:
            f_lev = gaussian_filter(fixed, sigma=sig)
            m_lev = gaussian_filter(moving, sigma=sig)
            f_lev = (f_lev - np.mean(f_lev)) / (np.std(f_lev) + 1e-6)
            m_lev = (m_lev - np.mean(m_lev)) / (np.std(m_lev) + 1e-6)

            def obj_func(p):
                angle, tx, ty = p
                #if abs(angle) > 0.8: return 1.0 
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s], [s, c]])
                off = center_rz - np.dot(R, center_rz) + np.array([tx, ty])
                warped = affine_transform(m_lev, R, offset=off, order=1, mode='nearest')
                
                std_warped = np.std(warped)
                if std_warped < 1e-6: return 0.0 
                
                corr = np.corrcoef(f_lev.ravel(), warped.ravel())[0, 1]
                return -corr if not np.isnan(corr) else 0.0

            res = minimize(obj_func, x0=current_p, method='Powell', tol=1e-3)
            current_p = res.x
        return current_p

    # --- INITIALISATION SÉCURISÉE DE LA PROGRESSION ---
    total_steps = (num_images - 1) + num_images
    current_step = 0
    progress = QProgressDialog("Recalage en cours...", "Annuler", 0, total_steps) if app else None
    
    if progress:
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

    # --- 2. BOUCLE DE CALCUL ---
    # Forward
    img_fixed = data_rz[:, :, img_ref_idx]
    for i in range(img_ref_idx, num_images - 1):
        if app:
            app.processEvents()
            if progress and progress.wasCanceled(): return None, {}
            
        params = get_transform_powell(img_fixed, data_rz[:, :, i+1])
        tform_params[i+1] = params
        
        c, s = np.cos(params[0]), np.sin(params[0])
        R = np.array([[c, -s], [s, c]])
        off = center_rz - np.dot(R, center_rz) + params[1:]
        img_fixed = affine_transform(data_rz[:, :, i+1], R, offset=off, order=1)
        
        current_step += 1
        if progress: progress.setValue(current_step)

    # Backward
    img_fixed = data_rz[:, :, img_ref_idx]
    for i in range(img_ref_idx, 0, -1):
        if app:
            app.processEvents()
            if progress and progress.wasCanceled(): return None, {}
            
        params = get_transform_powell(img_fixed, data_rz[:, :, i-1])
        tform_params[i-1] = params
        
        c, s = np.cos(params[0]), np.sin(params[0])
        R = np.array([[c, -s], [s, c]])
        off = center_rz - np.dot(R, center_rz) + params[1:]
        img_fixed = affine_transform(data_rz[:, :, i-1], R, offset=off, order=1)
        
        current_step += 1
        if progress: progress.setValue(current_step)

    # --- 3. LISSAGE TEMPOREL ---
    for col in range(3):
        tform_params[:, col] = gaussian_filter1d(tform_params[:, col], sigma=2.5)

    # --- 4. RECONSTRUCTION ---
    center_full = np.array([height / 2.0, width / 2.0])
    data_recal = np.zeros_like(data_repos)
    tform_results = {}

    for j in range(num_images):
        if app:
            app.processEvents()
            if progress and progress.wasCanceled(): return None, {} 
            
        angle, tx, ty = tform_params[j]
        tx_s, ty_s = tx / d, ty / d 
        
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        final_offset = center_full - np.dot(R, center_full) + np.array([tx_s, ty_s])
        
        data_recal[:, :, j] = affine_transform(data_repos[:, :, j], R, 
                                               offset=final_offset, 
                                               order=3, mode='constant', cval=0)
        
        T_mat = np.eye(3)
        T_mat[0,0], T_mat[0,1], T_mat[0,2] = c, s, tx_s
        T_mat[1,0], T_mat[1,1], T_mat[1,2] = -s, c, ty_s
        tform_results[j] = T_mat
        
        current_step += 1
        if progress: progress.setValue(current_step)

    data_recal = data_recal * mask_repaired[:, :, np.newaxis]
    if progress: progress.close()
    
    return data_recal.astype(np.uint8), tform_results
