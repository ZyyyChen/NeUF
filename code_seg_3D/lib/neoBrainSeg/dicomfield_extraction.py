<<<<<<< ours
import pydicom
from pathlib import Path

def dicomfield_extraction(main_dir, num_patient, ref_img_sag, ref_seq_dyn):
    """
    Extracts Ultrasound Region parameters from DICOM metadata.
    Note: Python lists are 0-indexed, so Item_1 becomes [0].
    """
    
    # 1. Clean the main path and point to the patient folder
    # resolve() fixes the "C:" drive letter and backslashes
    patient_dir = Path(main_dir.lstrip(':')).resolve() / 'Examen_echographie' / str(num_patient).strip()

    # 2. Build full file paths and ensure they end with .dcm
    path_seq = (patient_dir / str(ref_seq_dyn).strip()).with_suffix('.dcm')
    path_sag = (patient_dir / str(ref_img_sag).strip()).with_suffix('.dcm')

    print("path_sag", path_sag)
    print("path_seq", path_seq)

    # 3. Quick check: does the file actually exist?
    if not path_sag.exists():
        print(f"ERROR: File not found at {path_sag}")
    else:
        # 4. Read metadata (stop_before_pixels=True is like dicominfo)
        # str() ensures pydicom reads the path correctly on Windows
        ds_seqdyn = pydicom.dcmread(str(path_seq), stop_before_pixels=True)
        ds_sag = pydicom.dcmread(str(path_sag), stop_before_pixels=True)
        
        print("Success: Metadata loaded.")
    
    # Helper function to extract fields from the first Ultrasound Region
    def get_us_regions(ds):
        # SequenceOfUltrasoundRegions is a list of items
        region = ds.SequenceOfUltrasoundRegions[0]
        return (
            region.PhysicalDeltaX,
            region.RegionLocationMinY0,
            region.RegionLocationMaxY1,
            region.RegionLocationMinX0,
            region.RegionLocationMaxX1
        )
    
    # Extract for Sagittal image
    (delta_X_sag, min_X_sag, max_X_sag, min_Y_sag, max_Y_sag) = get_us_regions(ds_sag)
    # Extract for Dynamic Sequence
    (delta_X_seqdyn, min_X_seqdyn, max_X_seqdyn, min_Y_seqdyn, max_Y_seqdyn) = get_us_regions(ds_seqdyn)
    
    return (
        delta_X_sag, delta_X_seqdyn, 
        max_X_sag, max_X_seqdyn, 
        max_Y_sag, max_Y_seqdyn, 
        min_X_sag, min_X_seqdyn, 
        min_Y_sag, min_Y_seqdyn
    )
=======
import pydicom
from pathlib import Path

def dicomfield_extraction(main_dir, num_patient, ref_img_sag, ref_seq_dyn):
    """
    Extracts Ultrasound Region parameters from DICOM metadata.
    Note: Python lists are 0-indexed, so Item_1 becomes [0].
    """
    
    # 1. Clean the main path and point to the patient folder
    # resolve() fixes the "C:" drive letter and backslashes
    patient_dir = Path(main_dir.lstrip(':')).resolve() / 'Examen_echographie' / str(num_patient).strip()

    # 2. Build full file paths and ensure they end with .dcm
    path_seq = (patient_dir / str(ref_seq_dyn).strip()).with_suffix('.dcm')
    path_sag = (patient_dir / str(ref_img_sag).strip()).with_suffix('.dcm')

    print("path_sag", path_sag)
    print("path_seq", path_seq)

    # 3. Quick check: does the file actually exist?
    if not path_sag.exists():
        print(f"ERROR: File not found at {path_sag}")
    else:
        # 4. Read metadata (stop_before_pixels=True is like dicominfo)
        # str() ensures pydicom reads the path correctly on Windows
        ds_seqdyn = pydicom.dcmread(str(path_seq), stop_before_pixels=True)
        ds_sag = pydicom.dcmread(str(path_sag), stop_before_pixels=True)
        
        print("Success: Metadata loaded.")
    
    # Helper function to extract fields from the first Ultrasound Region
    def get_us_regions(ds):
        # SequenceOfUltrasoundRegions is a list of items
        region = ds.SequenceOfUltrasoundRegions[0]
        return (
            region.PhysicalDeltaX,
            region.RegionLocationMinY0,
            region.RegionLocationMaxY1,
            region.RegionLocationMinX0,
            region.RegionLocationMaxX1
        )
    
    # Extract for Sagittal image
    (delta_X_sag, min_X_sag, max_X_sag, min_Y_sag, max_Y_sag) = get_us_regions(ds_sag)
    # Extract for Dynamic Sequence
    (delta_X_seqdyn, min_X_seqdyn, max_X_seqdyn, min_Y_seqdyn, max_Y_seqdyn) = get_us_regions(ds_seqdyn)
    
    return (
        delta_X_sag, delta_X_seqdyn, 
        max_X_sag, max_X_seqdyn, 
        max_Y_sag, max_Y_seqdyn, 
        min_X_sag, min_X_seqdyn, 
        min_Y_sag, min_Y_seqdyn
    )
>>>>>>> theirs
