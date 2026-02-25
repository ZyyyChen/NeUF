import os
from PIL import Image
import pydicom
import numpy as np
from pydicom.dataset import FileDataset, Dataset
from pydicom.sequence import Sequence
import datetime
import argparse
from scipy.spatial.transform import Rotation as R

def multi_frame_dicom_with_position(input_folder, dicom_path, patient_id="02", patient_name="Test", height_mm=30, depth_mm=50, prefix="", suffix=".png"):
    """Generate a DICOM file from multiple images folder with 3D positions"""
    frames = []
    file_list = [prefix + str(num) + suffix for num in range(len(os.listdir(input_folder)))]
    
    for filename in file_list:
        image = Image.open(os.path.join(input_folder, filename)).convert('L')
        frames.append(np.array(image))

    height, width = frames[0].shape

    frames_np = np.stack(frames)  # (num_frames, height, width)

    file_meta = pydicom.Dataset()
    # file_meta.MediaStorageSOPClassUID = pydicom.uid.MultiFrameGrayscaleByteSecondaryCaptureImageStorage
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UltrasoundMultiFrameImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    dt = datetime.datetime.now()
    dicom = FileDataset(dicom_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    dicom.ContentDate = dt.strftime('%Y%m%d') 
    dicom.ContentTime = dt.strftime('%H%M%S.%f')

    dicom.PatientName = patient_name
    dicom.PatientID = patient_id

    dicom.Modality = "US"
    dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    dicom.StudyInstanceUID = pydicom.uid.generate_uid()
    dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID

    dicom.NumberOfFrames = frames_np.shape[0]
    dicom.Rows = frames_np.shape[1]
    dicom.Columns = frames_np.shape[2]
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.BitsAllocated = 8
    dicom.BitsStored = 8
    dicom.HighBit = 7
    dicom.PixelRepresentation = 0
    dicom.PixelData = frames_np.tobytes()

    # Set Pixel Spacing (mm) and Slice Thickness
    dicom.PixelSpacing = [height_mm/height, height_mm/height]  # Row spacing, Column spacing in mm
    step = depth_mm / len(file_list)
    dicom.SliceThickness = step
    dicom.SpacingBetweenSlices = step

    # Shared Functional Groups
    shared_fg = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 0, 1]    # Y-axis oriented
    shared_fg.PlaneOrientationSequence = Sequence([orientation])
    dicom.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-frame Functional Groups
    dicom.PerFrameFunctionalGroupsSequence = Sequence()
    for i in range(frames_np.shape[0]):
        fg = Dataset()
        position = Dataset()
        pos = 0.0 + i * step
        position.ImagePositionPatient = [0.0, pos , 0.0]  # X, Y, Z position of the slice
        fg.PlanePositionSequence = Sequence([position])
        dicom.PerFrameFunctionalGroupsSequence.append(fg)

    dicom.save_as(dicom_path, write_like_original=False)
    print(f"Saved multi-frame 3D DICOM: {dicom_path}")

def multi_frame_dicom_with_infosdat_positions(input_folder, dicom_path, infos_file_path,
                                              patient_id="02", patient_name="Test",
                                              prefix="", suffix=".png", crop=None):
    """Generate a DICOM file with accurate position + orientation from infos.dat"""

    # Load image filenames
    file_list = [prefix + str(num) + suffix for num in range(len(os.listdir(input_folder)))]
    if crop:
        crop_tuple = (crop['x'],crop['y'],crop['x']+crop['width'],crop['y']+crop['height'])
        frames = [np.array(Image.open(os.path.join(input_folder, f)).crop(crop_tuple).convert('L')) for f in file_list]
    else:
        frames = [np.array(Image.open(os.path.join(input_folder, f)).convert('L')) for f in file_list]
    frames_np = np.stack(frames)  # shape: (num_frames, height, width)

    height, width = frames_np.shape[1:]
    
    # Load infos.dat (skip header)
    with open(infos_file_path, "r") as f:
        lines = f.readlines()[1:]

    positions = np.array([list(map(float, l.strip().split()[:3])) for l in lines])
    quaternions = np.array([list(map(float, l.strip().split()[3:7])) for l in lines])
    width_mm, depth_mm = map(float, lines[0].strip().split()[-2:])

    # Flip quaternion vector components (convert to ITK-style if needed)
    quaternions[:, 1:] *= -1  # Optional: depends on how original data was recorded

    # Prepare DICOM metadata
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UltrasoundMultiFrameImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    dt = datetime.datetime.now()
    dicom = FileDataset(dicom_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient/study metadata
    dicom.ContentDate = dt.strftime('%Y%m%d')
    dicom.ContentTime = dt.strftime('%H%M%S.%f')
    dicom.PatientName = patient_name
    dicom.PatientID = patient_id
    dicom.Modality = "US"
    dicom.StudyInstanceUID = pydicom.uid.generate_uid()
    dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Image metadata
    dicom.NumberOfFrames = len(frames)
    dicom.Rows = height
    dicom.Columns = width
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.BitsAllocated = 8
    dicom.BitsStored = 8
    dicom.HighBit = 7
    dicom.PixelRepresentation = 0
    dicom.PixelData = frames_np.tobytes()

    # Pixel spacing
    dicom.PixelSpacing = [depth_mm / height, width_mm / width]  # [row_spacing, col_spacing]
    dicom.SliceThickness = 1.0
    dicom.SpacingBetweenSlices = 1.0

    # Shared Functional Groups (can be minimal if overridden per-frame)
    dicom.SharedFunctionalGroupsSequence = Sequence()

    # Per-frame Functional Groups
    dicom.PerFrameFunctionalGroupsSequence = Sequence()
    for i, quat in enumerate(quaternions):
        fg = Dataset()

        # Plane Position
        position = Dataset()
        position.ImagePositionPatient = [float(p) for p in positions[i]]
        fg.PlanePositionSequence = Sequence([position])

        # Orientation (convert quaternion -> direction cosines)
        rot = R.from_quat(quat)  # (x, y, z, w) assumed; reverse if (w, x, y, z)
        R_matrix = rot.as_matrix()  # 3x3
        row_dir = R_matrix[:, 0]
        col_dir = R_matrix[:, 1]
        orientation = Dataset()
        orientation.ImageOrientationPatient = [*row_dir, *col_dir]
        fg.PlaneOrientationSequence = Sequence([orientation])

        dicom.PerFrameFunctionalGroupsSequence.append(fg)

    # Save the DICOM
    dicom.save_as(dicom_path, write_like_original=False)
    print(f"DICOM saved to: {dicom_path}")

def images2gif(input_folder, output_gif, duration=100, prefix="", suffix=".png", crop=None):
    """Convert a folder of images in an animated .GIF"""
    images = []
    num_its = sorted([int(f.split(".")[0]) for f in os.listdir(input_folder)])
    for num in num_its:
        filename = prefix + str(num) + suffix
        if crop: images.append(Image.open(os.path.join(input_folder, filename)).crop(crop).convert('L').resize((400,400)))
        else: images.append(Image.open(os.path.join(input_folder, filename)).convert('L').resize((400,400)))
    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        default='./volume/Generated/tests/ckpt/',
                        help='folder containing ckpt files',
                        required=True)
    args = parser.parse_args()
    
    # reco_folder = os.path.join(args.input_dir, "us")    # folder with the reproduced dataset images
    # output_dicom = os.path.join(args.input_dir,"test.dcm")
    
    # crop_info = {"01":  (2* 202 , 2 * 4      ,(202+310)*2    ,(4  +178)*2      ),
    #              "03":  (2* 253 , 2 * 12     ,(253+335)*2    ,(12 +213)*2     ),
    #              "07":  (2* 344 , 2 * 31     ,(344+198)*2    ,(31 +192)*2     ),
    #              "11":  (2* 142 , 2 * 183    ,(142+182)*2    ,(183+171)*2    )
    #             }
    num_slice = "200"
    subfolders = [ f.path for f in os.scandir(args.input_dir) if f.is_dir() ]
    for folder in subfolders:
        images2gif(input_folder=os.path.join(folder,f"slice_{num_slice}"), 
                   output_gif= os.path.join(args.input_dir, os.path.basename(folder) + f"slice_{num_slice}" ".gif"),
                   duration=250,
                   prefix="",
                   suffix=".png",
                   crop=None)#crop_info[os.path.basename(folder).split("_full")[0].split("_")[1]])
    
    # multi_frame_dicom_with_position(input_folder=reco_folder,
    #                                 dicom_path=output_dicom)