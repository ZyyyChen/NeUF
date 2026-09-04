<<<<<<< ours
import numpy as np
import dtcwt

"""def extend_denoise_2d(data):
    h, w = data.shape
    new_h = 2**int(np.ceil(np.log2(h)))
    new_w = 2**int(np.ceil(np.log2(w)))
    
    datab = np.zeros((new_h, new_w), dtype=data.dtype)
    datab[:h, :w] = data
    
    dx = new_h - h
    dy = new_w - w
    
    if dx >= 1:
        # Réplication des lignes
        datab[h:, :w] = data[:dx, :]
    if dy >= 1:
        # Réplication des colonnes
        datab[:h, w:] = data[:, :dy]
    if dx >= 1 and dy >= 1:
        # Réplication du coin
        datab[h:, w:] = data[:dx, :dy]
        
    return datab, (h, w)

def denoise_data_2d(data, T):
   
    # 1. Extension de la taille (Power of 2)
    data_ex, original_shape = extend_denoise_2d(data)
    
    # 2. Transformation DT-CWT 2D (Équivalent denC2D)
    # J=4 comme spécifié dans votre code MATLAB
    transform = dtcwt.Transform2d()
    t = transform.forward(data_ex, nlevels=4)
    
    # 3. Seuillage doux (Soft Thresholding)
    # En Python, dtcwt gère déjà les coefficients complexes (Real + I*Imag)
    new_highpasses = []
    for j in range(len(t.highpasses)):
        coeffs = t.highpasses[j]
        
        # Calcul de la magnitude
        magnitude = np.abs(coeffs)
        
        # Application du gain (Soft thresholding)
        # gain = max(1 - T/magnitude, 0)
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        
        new_highpasses.append(coeffs * gain)
        
    # 4. Reconstruction (Inverse Transform)
    t.highpasses = tuple(new_highpasses)
    data_dnsd = transform.inverse(t)
    
    # 5. Retour à la taille originale (set_size_of_to)
    return data_dnsd[:original_shape[0], :original_shape[1]].astype(np.float32)

def set_size_of_to(data_in, array_size):

    h_out, w_out = array_size[:2]
    h_in, w_in = data_in.shape[:2]
    
    out = np.zeros((h_out, w_out), dtype=data_in.dtype)
    
    # Indices communs
    ih = min(h_out, h_in)
    iw = min(w_out, w_in)
    
    out[:ih, :iw] = data_in[:ih, :iw]
    
    # Réplication pour remplir si la sortie est plus grande
    if h_out > h_in:
        for j in range(h_in, h_out):
            out[j, :iw] = data_in[h_in-1, :iw]
            
    if w_out > w_in:
        for j in range(w_in, w_out):
            out[:, j] = out[:, w_in-1]
            
    return out"""


import numpy as np
import dtcwt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import os

def _worker_denoise_final(slice_2d, T):
    """
    DTCWT 2D optimisée : Normalisation + Padding Réfléchi + Masquage de fond.
    Élimine les artefacts blancs visibles aux bords.
    """
    # --- 1. SAUVEGARDE ET NORMALISATION ---
    f_min, f_max = np.min(slice_2d), np.max(slice_2d)
    if f_max <= f_min:
        return slice_2d.astype(np.float32)
    
    # Normalisation 0-255 pour que le seuil T=40 soit efficace
    slice_norm = 255.0 * (slice_2d - f_min) / (f_max - f_min)

    # --- 2. PADDING SYMÉTRIQUE (ANTI-BORDS BLANCS) ---
    h, w = slice_norm.shape
    margin = 32  # Marge de sécurité pour absorber les instabilités
    
    def get_pad_params(size, m):
        total = size + 2 * m
        extra = (16 - total % 16) % 16
        return m, m + extra

    ph_top, ph_bottom = get_pad_params(h, margin)
    pw_left, pw_right = get_pad_params(w, margin)
    
    # 'reflect' crée une continuité qui empêche les pics blancs
    data_ex = np.pad(slice_norm, ((ph_top, ph_bottom), (pw_left, pw_right)), mode='reflect')

    # --- 3. TRANSFORMATION ET SEUILLAGE ---
    transform = dtcwt.Transform2d()
    t = transform.forward(data_ex, nlevels=4) # J=4
    
    new_highpasses = []
    for coeffs in t.highpasses:
        mag = np.abs(coeffs)
        gain = np.zeros_like(mag)
        # Division sécurisée pour éviter les RuntimeWarnings
        np.divide(T, mag, out=gain, where=mag > 1e-5)
        gain = np.maximum(1 - gain, 0)
        new_highpasses.append(coeffs * gain)
        
    t.highpasses = tuple(new_highpasses)
    res = transform.inverse(t)
    
    # --- 4. RECADRAGE ET NETTOYAGE FINAL ---
    # Extraction de la zone utile (supprime la marge polluée)
    res_crop = res[ph_top:ph_top+h, pw_left:pw_left+w]
    
    # Masquage optionnel : Si le pixel original était très sombre, on force le noir
    # Cela aide à nettoyer les bords si T est trop grand
    background_mask = (slice_norm > 5) # Seuil bas sur l'image normalisée
    res_crop = res_crop * background_mask
    
    # Dénormalisation
    res_final = (res_crop / 255.0) * (f_max - f_min) + f_min
    
    return res_final.astype(np.float32)

def denoise_volume_2d_parallel(volume, T=40):
    if volume is None:
        return None
    
    num_slices = volume.shape[2]
    slices = [volume[:, :, i] for i in range(num_slices)]
    
    worker_func = partial(_worker_denoise_final, T=T)
    print(f"Débruitage sécurisé (T={T}) sur {num_slices} coupes...")
    
    # Utilisation d'un contexte explicite pour éviter les fuites de handles
    ctx = multiprocessing.get_context('spawn')
    
    # Limite le nombre de workers pour éviter de saturer la RAM (ex: 227 coupes)
    max_workers = min(os.cpu_count() or 1, 4) 
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        results = list(executor.map(worker_func, slices))

    return np.stack(results, axis=2)
=======
import numpy as np
import dtcwt

"""def extend_denoise_2d(data):
    h, w = data.shape
    new_h = 2**int(np.ceil(np.log2(h)))
    new_w = 2**int(np.ceil(np.log2(w)))
    
    datab = np.zeros((new_h, new_w), dtype=data.dtype)
    datab[:h, :w] = data
    
    dx = new_h - h
    dy = new_w - w
    
    if dx >= 1:
        # Réplication des lignes
        datab[h:, :w] = data[:dx, :]
    if dy >= 1:
        # Réplication des colonnes
        datab[:h, w:] = data[:, :dy]
    if dx >= 1 and dy >= 1:
        # Réplication du coin
        datab[h:, w:] = data[:dx, :dy]
        
    return datab, (h, w)

def denoise_data_2d(data, T):
   
    # 1. Extension de la taille (Power of 2)
    data_ex, original_shape = extend_denoise_2d(data)
    
    # 2. Transformation DT-CWT 2D (Équivalent denC2D)
    # J=4 comme spécifié dans votre code MATLAB
    transform = dtcwt.Transform2d()
    t = transform.forward(data_ex, nlevels=4)
    
    # 3. Seuillage doux (Soft Thresholding)
    # En Python, dtcwt gère déjà les coefficients complexes (Real + I*Imag)
    new_highpasses = []
    for j in range(len(t.highpasses)):
        coeffs = t.highpasses[j]
        
        # Calcul de la magnitude
        magnitude = np.abs(coeffs)
        
        # Application du gain (Soft thresholding)
        # gain = max(1 - T/magnitude, 0)
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        
        new_highpasses.append(coeffs * gain)
        
    # 4. Reconstruction (Inverse Transform)
    t.highpasses = tuple(new_highpasses)
    data_dnsd = transform.inverse(t)
    
    # 5. Retour à la taille originale (set_size_of_to)
    return data_dnsd[:original_shape[0], :original_shape[1]].astype(np.float32)

def set_size_of_to(data_in, array_size):

    h_out, w_out = array_size[:2]
    h_in, w_in = data_in.shape[:2]
    
    out = np.zeros((h_out, w_out), dtype=data_in.dtype)
    
    # Indices communs
    ih = min(h_out, h_in)
    iw = min(w_out, w_in)
    
    out[:ih, :iw] = data_in[:ih, :iw]
    
    # Réplication pour remplir si la sortie est plus grande
    if h_out > h_in:
        for j in range(h_in, h_out):
            out[j, :iw] = data_in[h_in-1, :iw]
            
    if w_out > w_in:
        for j in range(w_in, w_out):
            out[:, j] = out[:, w_in-1]
            
    return out"""


import numpy as np
import dtcwt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import os

def _worker_denoise_final(slice_2d, T):
    """
    DTCWT 2D optimisée : Normalisation + Padding Réfléchi + Masquage de fond.
    Élimine les artefacts blancs visibles aux bords.
    """
    # --- 1. SAUVEGARDE ET NORMALISATION ---
    f_min, f_max = np.min(slice_2d), np.max(slice_2d)
    if f_max <= f_min:
        return slice_2d.astype(np.float32)
    
    # Normalisation 0-255 pour que le seuil T=40 soit efficace
    slice_norm = 255.0 * (slice_2d - f_min) / (f_max - f_min)

    # --- 2. PADDING SYMÉTRIQUE (ANTI-BORDS BLANCS) ---
    h, w = slice_norm.shape
    margin = 32  # Marge de sécurité pour absorber les instabilités
    
    def get_pad_params(size, m):
        total = size + 2 * m
        extra = (16 - total % 16) % 16
        return m, m + extra

    ph_top, ph_bottom = get_pad_params(h, margin)
    pw_left, pw_right = get_pad_params(w, margin)
    
    # 'reflect' crée une continuité qui empêche les pics blancs
    data_ex = np.pad(slice_norm, ((ph_top, ph_bottom), (pw_left, pw_right)), mode='reflect')

    # --- 3. TRANSFORMATION ET SEUILLAGE ---
    transform = dtcwt.Transform2d()
    t = transform.forward(data_ex, nlevels=4) # J=4
    
    new_highpasses = []
    for coeffs in t.highpasses:
        mag = np.abs(coeffs)
        gain = np.zeros_like(mag)
        # Division sécurisée pour éviter les RuntimeWarnings
        np.divide(T, mag, out=gain, where=mag > 1e-5)
        gain = np.maximum(1 - gain, 0)
        new_highpasses.append(coeffs * gain)
        
    t.highpasses = tuple(new_highpasses)
    res = transform.inverse(t)
    
    # --- 4. RECADRAGE ET NETTOYAGE FINAL ---
    # Extraction de la zone utile (supprime la marge polluée)
    res_crop = res[ph_top:ph_top+h, pw_left:pw_left+w]
    
    # Masquage optionnel : Si le pixel original était très sombre, on force le noir
    # Cela aide à nettoyer les bords si T est trop grand
    background_mask = (slice_norm > 5) # Seuil bas sur l'image normalisée
    res_crop = res_crop * background_mask
    
    # Dénormalisation
    res_final = (res_crop / 255.0) * (f_max - f_min) + f_min
    
    return res_final.astype(np.float32)

def denoise_volume_2d_parallel(volume, T=40):
    if volume is None:
        return None
    
    num_slices = volume.shape[2]
    slices = [volume[:, :, i] for i in range(num_slices)]
    
    worker_func = partial(_worker_denoise_final, T=T)
    print(f"Débruitage sécurisé (T={T}) sur {num_slices} coupes...")
    
    # Utilisation d'un contexte explicite pour éviter les fuites de handles
    ctx = multiprocessing.get_context('spawn')
    
    # Limite le nombre de workers pour éviter de saturer la RAM (ex: 227 coupes)
    max_workers = min(os.cpu_count() or 1, 4) 
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        results = list(executor.map(worker_func, slices))

    return np.stack(results, axis=2)
>>>>>>> theirs
