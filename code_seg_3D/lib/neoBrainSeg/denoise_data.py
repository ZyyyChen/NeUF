<<<<<<< ours
import numpy as np
import dtcwt
import time

def pad_to_multiple(data, factor=16):
    """Ajoute des zéros pour que chaque dimension soit un multiple de factor."""
    h, w, d = data.shape
    
    # Calcul des nouveaux bords
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    d_pad = (factor - d % factor) % factor
    
    # Padding (ajout de zéros à la fin de chaque axe)
    data_padded = np.pad(data, ((0, h_pad), (0, w_pad), (0, d_pad)), mode='constant')
    
    return data_padded, (h, w, d)

def unpad_to_original(data, original_shape):
    """Retire le padding pour revenir à la taille initiale."""
    h, w, d = original_shape
    return data[:h, :w, :d]

def denoise_data(data, T):
    # 1. Sauvegarde de la taille originale et Padding
    # On utilise 16 car J=4 niveaux nécessite 2^4
    data_padded, original_shape = pad_to_multiple(data, factor=16)
    
    # 2. Débruitage sur le volume aux bonnes dimensions
    transform = dtcwt.Transform3d()
    t = transform.forward(data_padded, nlevels=4) # J=4 comme dans votre MATLAB
    
    new_highpasses = []
    for coeffs in t.highpasses:
        magnitude = np.abs(coeffs)
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        new_highpasses.append(coeffs * gain)
        
    t.highpasses = tuple(new_highpasses)
    data_dnsd = transform.inverse(t)
    
    # 3. Retour à la taille initiale (équivalent de tools.set_size_of_to)
    return unpad_to_original(data_dnsd, original_shape).astype(np.float32)

def denC3D_python(x, T):
    """
    Traduction de la fonction denC3D (Seuillage doux en 3D)
    """
    # Initialisation de la transformée 3D
    # Par défaut, dtcwt utilise les filtres de Farras pour le niveau 1 
    # et Kingsbury pour les suivants (J=4).
    transform = dtcwt.Transform3d()
    J = 4
    
    # Forward Transform (équivalent de cplxdual3D)
    t = transform.forward(x, nlevels=J)
    
    # Filtrage des coefficients (Soft Thresholding)
    # w{j}{m}{n}{p}{d} en MATLAB devient t.highpasses[j] en Python
    new_highpasses = []
    
    for j in range(J):
        # En Python, les coefficients sont déjà stockés sous forme complexe 
        # (C = Real + I*Imag) dans les highpasses.
        coeffs = t.highpasses[j]
        
        # Application du seuillage doux (soft thresholding)
        magnitude = np.abs(coeffs)
        # On évite la division par zéro avec np.where
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        
        new_highpasses.append(coeffs * gain)
        
    # Reconstruction (équivalent de icplxdual3D)
    t.highpasses = tuple(new_highpasses)
    y = transform.inverse(t)
    
    return y.astype(np.float32)

def soft_threshold(C, T):
    """
    Implémentation manuelle du seuillage doux pour des nombres complexes.
    C = C * max(1 - T/abs(C), 0)
    """
    mag = np.abs(C)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(mag > 0, C * np.maximum(1 - T/mag, 0), 0)
    return res
=======
import numpy as np
import dtcwt
import time

def pad_to_multiple(data, factor=16):
    """Ajoute des zéros pour que chaque dimension soit un multiple de factor."""
    h, w, d = data.shape
    
    # Calcul des nouveaux bords
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    d_pad = (factor - d % factor) % factor
    
    # Padding (ajout de zéros à la fin de chaque axe)
    data_padded = np.pad(data, ((0, h_pad), (0, w_pad), (0, d_pad)), mode='constant')
    
    return data_padded, (h, w, d)

def unpad_to_original(data, original_shape):
    """Retire le padding pour revenir à la taille initiale."""
    h, w, d = original_shape
    return data[:h, :w, :d]

def denoise_data(data, T):
    # 1. Sauvegarde de la taille originale et Padding
    # On utilise 16 car J=4 niveaux nécessite 2^4
    data_padded, original_shape = pad_to_multiple(data, factor=16)
    
    # 2. Débruitage sur le volume aux bonnes dimensions
    transform = dtcwt.Transform3d()
    t = transform.forward(data_padded, nlevels=4) # J=4 comme dans votre MATLAB
    
    new_highpasses = []
    for coeffs in t.highpasses:
        magnitude = np.abs(coeffs)
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        new_highpasses.append(coeffs * gain)
        
    t.highpasses = tuple(new_highpasses)
    data_dnsd = transform.inverse(t)
    
    # 3. Retour à la taille initiale (équivalent de tools.set_size_of_to)
    return unpad_to_original(data_dnsd, original_shape).astype(np.float32)

def denC3D_python(x, T):
    """
    Traduction de la fonction denC3D (Seuillage doux en 3D)
    """
    # Initialisation de la transformée 3D
    # Par défaut, dtcwt utilise les filtres de Farras pour le niveau 1 
    # et Kingsbury pour les suivants (J=4).
    transform = dtcwt.Transform3d()
    J = 4
    
    # Forward Transform (équivalent de cplxdual3D)
    t = transform.forward(x, nlevels=J)
    
    # Filtrage des coefficients (Soft Thresholding)
    # w{j}{m}{n}{p}{d} en MATLAB devient t.highpasses[j] en Python
    new_highpasses = []
    
    for j in range(J):
        # En Python, les coefficients sont déjà stockés sous forme complexe 
        # (C = Real + I*Imag) dans les highpasses.
        coeffs = t.highpasses[j]
        
        # Application du seuillage doux (soft thresholding)
        magnitude = np.abs(coeffs)
        # On évite la division par zéro avec np.where
        gain = np.zeros_like(magnitude)
        np.divide(T, magnitude, out=gain, where=magnitude > 0)
        gain = np.maximum(1 - gain, 0)
        
        new_highpasses.append(coeffs * gain)
        
    # Reconstruction (équivalent de icplxdual3D)
    t.highpasses = tuple(new_highpasses)
    y = transform.inverse(t)
    
    return y.astype(np.float32)

def soft_threshold(C, T):
    """
    Implémentation manuelle du seuillage doux pour des nombres complexes.
    C = C * max(1 - T/abs(C), 0)
    """
    mag = np.abs(C)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(mag > 0, C * np.maximum(1 - T/mag, 0), 0)
    return res
>>>>>>> theirs
