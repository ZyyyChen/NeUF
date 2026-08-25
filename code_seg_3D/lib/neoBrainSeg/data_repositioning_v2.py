from tools.MITKviewer import valider_recalage

def data_repositioning_v2(
        data_sag, data_seqdyn,
        max_X_sag, max_X_seqdyn,
        max_Y_sag, max_Y_seqdyn,
        min_X_sag, min_X_seqdyn,
        min_Y_sag, min_Y_seqdyn,
        ref):

    # -------------------------------------------------
    # MATLAB → Python index rules reminder:
    #
    # MATLAB: 1..N inclusive
    # Python: 0..N-1 ; slicing end is exclusive
    #
    # MATLAB A(1:250) → Python A[0:250]
    # MATLAB A(35:70) → Python A[34:70]
    # MATLAB A(end) → Python A[-1] or :
    # MATLAB A(x:end) → Python A[x-1:]
    #
    # -------------------------------------------------

    # -------- Copy --------
    data_clean_sag = data_sag.copy()
    data_clean_seqdyn = data_seqdyn.copy()

    # =================================================
    # CLEAN SAG ---------------- (MATLAB was 2D but allowed 3D syntax)
    # =================================================
    if data_clean_sag.ndim == 2:
        data_clean_sag[:min_X_sag, :] = 0
        data_clean_sag[max_X_sag-1:, :] = 0
        data_clean_sag[:, :min_Y_sag] = 0
        data_clean_sag[:, max_Y_sag-11:] = 0
    else:
        data_clean_sag[:min_X_sag, :, :] = 0
        data_clean_sag[max_X_sag-1:, :, :] = 0
        data_clean_sag[:, :min_Y_sag, :] = 0
        data_clean_sag[:, max_Y_sag-11:, :] = 0
    
    # ---------------- special SAG ----------------
    if ref == 'PatientPureWave_JY_Z':
        data_clean_sag[0:250, 0:125] = 0
        data_clean_sag[34:70, 249:295] = 0
        data_clean_sag[29:50, 819:990] = 0
        data_clean_sag[64:495, 875:1000] = 0
        data_clean_sag[514:760, 910:930] = 0
        data_clean_sag[749:765, 850:925] = 0

    elif ref == 'PatientETO_J1_2':
        data_clean_sag[0:153, 0:86] = 0
        data_clean_sag[58:82, 202:234] = 0

    elif ref == 'Patient02_J2_2':
        data_clean_sag[0:260, 885:] = 0

    elif ref in ('Patient02_J2_3', 'Patient02_J2_4'):
        data_clean_sag[0:257, 892:] = 0

    elif ref.startswith('Fantome_J1') or ref.startswith('Fantome02_J1'):
        data_clean_sag[99:, 29:65] = 0
        data_clean_sag[99:120, 254:275] = 0
    
    elif ref == 'Patient36_J12_46':
        data_clean_sag[0:12, :] = 0 
    elif ref == 'Patient2_J0_0':
        data_clean_sag[:, 896:] = 0
        data_clean_sag[:, :100] = 0
    else:
        data_clean_sag[0:150, 0:100] = 0
        data_clean_sag[49:85, 200:230] = 0

    # =================================================
    # CLEAN SEQ-DYN  (always 3D)
    # =================================================
    data_clean_seqdyn[:min_X_seqdyn, :, :] = 0
    data_clean_seqdyn[max_X_seqdyn-1:, :, :] = 0
    data_clean_seqdyn[:, max_Y_seqdyn-11:, :] = 0
    print(min_X_seqdyn, max_X_seqdyn)
    print(min_Y_seqdyn, max_Y_seqdyn)

    # ---------------- special SEQ ----------------
    if ref == 'PatientPureWave_JY_Z':
        data_clean_seqdyn[0:200, 0:100] = 0
        data_clean_seqdyn[24:60, 170:205] = 0
        data_clean_seqdyn[519:540, 130:210] = 0
        data_clean_seqdyn[19:35, 640:770] = 0
        data_clean_seqdyn[24:280, 690:790] = 0
        data_clean_seqdyn[299:600, 715:790] = 0
        data_clean_seqdyn[569:, 660:, :] = 0

    elif ref == 'PatientETO_J1_2':
        data_clean_seqdyn[0:364, 0:158] = 0
        data_clean_seqdyn[16:60, 242:479] = 0
        data_clean_seqdyn[0:269, 754:] = 0
        data_clean_seqdyn[709:, :, :] = 0

    elif ref == 'Patient02_J2_2':
        data_clean_seqdyn[0:83, 0:165] = 0
        data_clean_seqdyn[0:260, 882:] = 0

    elif ref in ('Patient02_J2_3', 'Patient02_J2_4'):
        data_clean_seqdyn[0:257, 892:] = 0

    elif ref.startswith('Fantome_J1'):
        data_clean_seqdyn[:, 0:65, :] = 0
        data_clean_seqdyn[99:120, 254:275] = 0
    elif ref == 'Patient2_J0_0':
        print("Special case: Patient2_J0_0 - applying large mask to SEQ-DYN")
        #data_clean_seqdyn[0:150, 0:100] = 0
        #data_clean_seqdyn[49:85, 160:] = 0
        data_clean_seqdyn[:, 693:] = 0
        data_clean_seqdyn[:, :100] = 0
        print(data_clean_seqdyn.shape)
        pass
    else:
        data_clean_seqdyn[0:150, 0:100] = 0
        data_clean_seqdyn[49:85, 200:230] = 0


    #Ñimport nrrd
    #nrrd.write('data_clean_seqdyn.nrrd', data_clean_seqdyn)
    #nrrd.write('data_seqdyn.nrrd', data_seqdyn)
    #nrrd.write('data_clean_sag.nrrd', data_clean_sag)
    #nrrd.write('data_sag.nrrd', data_sag)
    

    # =================================================
    # ROI REPOSITIONING — exact MATLAB behavior
    # =================================================
    data_repos_sag = data_clean_sag[min_X_sag-1:max_X_sag,
                                    min_Y_sag-1:max_Y_sag]

    data_repos_seqdyn = data_clean_seqdyn[min_X_seqdyn-1:max_X_seqdyn,
                                          min_Y_seqdyn-1:max_Y_seqdyn, :]
    
    print(data_sag.shape)
    print(data_clean_sag.shape, data_repos_sag.shape)
    print(data_seqdyn.shape)
    print(data_clean_seqdyn.shape, data_repos_seqdyn.shape)

    # Fantome extra crop
    if ref.startswith('Fantome_J1'):
        data_repos_seqdyn = data_repos_seqdyn[35:460, 50:600, :]
        data_repos_sag = data_repos_sag[35:460, 50:600]

    # =================================================
    # HUGE SPECIAL CASE: Patient36_J12_46
    # =================================================
    if ref == 'Patient36_J12_46':
        d = 3

        # top strip
        data_repos_seqdyn[0:2, :, :] = 0

        def kill(x, y):
            data_repos_seqdyn[max(x-d,0):x+d+1,
                              max(y-d,0):y+d+1, :] = 0

        # left arc
        pts_left = [
            (1,247),(22,233),(52,217),(82,200),(112,184),(141,167),
            (171,151),(201,134),(231,118),(260,101),(290,85),
            (319,68),(349,52),(379,35),(408,19),(438,2)
        ]
        for x,y in pts_left:
            kill(x,y)

        # right arc
        pts_right = [
            (3,509),(33,526),(62,542),(92,559),(122,575),(152,592),
            (181,608),(211,625),(241,641),(271,657),(300,674),
            (330,691),(359,707),(389,724),(418,739)
        ]
        for x,y in pts_right:
            kill(x,y)

        # bottom arc
        pts_bottom = [
            (479,12),(491,37),(502,63),(513,88),(522,114),
            (531,141),(539,168),(547,195),(552,222),(557,250),
            (561,278),(564,305),
            (566,333),(566,361),(566,390),(566,417),
            (564,445),(561,473),(557,501),(552,528),
            (546,556),(540,583),(532,610),(524,636),
            (515,662),(504,688),(493,714),(481,739)
        ]
        for x,y in pts_bottom:
            kill(x,y)
    
    if 0 in data_repos_seqdyn.shape or 0 in data_repos_sag.shape:
        print("Warning: data_repos_seqdyn or data_repos_sag has a zero dimension, likely due to aggressive cropping. Check the min/max coordinates and special cases in dicom tags.")
        data_repos_sag = data_sag[:,int(data_sag.shape[1]/4):data_sag.shape[1]-int(data_sag.shape[1]/4)]  # fallback to original if cropping failed
        data_repos_seqdyn = data_seqdyn[:,int(data_seqdyn.shape[1]/4):data_seqdyn.shape[1]-int(data_seqdyn.shape[1]/4), :]  # fallback to original if cropping failed

    #print("Debug: data_repos_seqdyn shape:", data_repos_seqdyn.shape)
    #print("Debug: data_repos_sag shape:", data_repos_sag.shape)
    #import nrrd
    #nrrd.write('data_repos_seqdyn.nrrd', data_repos_seqdyn)
    #nrrd.write('data_repos_sag.nrrd', data_repos_sag)
    return data_repos_sag, data_repos_seqdyn
