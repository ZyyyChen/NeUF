<<<<<<< ours
def ref_parameters_extraction_v3(idx, params_patients_echo, ref_patients_echo):
    """
    Extracts reference data and parameters used for 3D reconstruction.
    
    """
    
    # --- Parameters from ref_patients_echo ---
    # Reference of the dynamic sequence
    ref = ref_patients_echo[idx][0] 
    
    # Extract Patient Number: finds string before the first underscore
    if '_' in ref:
        num_patient = ref.split('_')[0]
    else:
        num_patient = ref

    ref_seq_dyn = ref_patients_echo[idx][1]   # Ref for dynamic sequence
    ref_img_sag = ref_patients_echo[idx][2]   # Ref for sagittal image
    acq_plan    = ref_patients_echo[idx][3]
    acq_dir     = ref_patients_echo[idx][4]   # Sweep direction
    
    # --- Parameters from reconstruction parameters ---
    
    # Origin coordinates
    origin_coord = [
        params_patients_echo[idx][0], 
        params_patients_echo[idx][1], 
        params_patients_echo[idx][2]
    ]

    angle_rot_cor  = params_patients_echo[idx][3]
    angle_rot_sag  = params_patients_echo[idx][4]
    delta_X_seqdyn = params_patients_echo[idx][5]
    delta_X_cc     = params_patients_echo[idx][6]

    # Return as a tuple to match MATLAB function signature
    return (
        acq_dir, acq_plan, angle_rot_cor, angle_rot_sag, delta_X_cc, 
        delta_X_seqdyn, num_patient, origin_coord, 
        ref, ref_seq_dyn, ref_img_sag
    )
=======
def ref_parameters_extraction_v3(idx, params_patients_echo, ref_patients_echo):
    """
    Extracts reference data and parameters used for 3D reconstruction.
    
    """
    
    # --- Parameters from ref_patients_echo ---
    # Reference of the dynamic sequence
    ref = ref_patients_echo[idx][0] 
    
    # Extract Patient Number: finds string before the first underscore
    if '_' in ref:
        num_patient = ref.split('_')[0]
    else:
        num_patient = ref

    ref_seq_dyn = ref_patients_echo[idx][1]   # Ref for dynamic sequence
    ref_img_sag = ref_patients_echo[idx][2]   # Ref for sagittal image
    acq_plan    = ref_patients_echo[idx][3]
    acq_dir     = ref_patients_echo[idx][4]   # Sweep direction
    
    # --- Parameters from reconstruction parameters ---
    
    # Origin coordinates
    origin_coord = [
        params_patients_echo[idx][0], 
        params_patients_echo[idx][1], 
        params_patients_echo[idx][2]
    ]

    angle_rot_cor  = params_patients_echo[idx][3]
    angle_rot_sag  = params_patients_echo[idx][4]
    delta_X_seqdyn = params_patients_echo[idx][5]
    delta_X_cc     = params_patients_echo[idx][6]

    # Return as a tuple to match MATLAB function signature
    return (
        acq_dir, acq_plan, angle_rot_cor, angle_rot_sag, delta_X_cc, 
        delta_X_seqdyn, num_patient, origin_coord, 
        ref, ref_seq_dyn, ref_img_sag
    )
>>>>>>> theirs
