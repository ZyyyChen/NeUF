import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MultiFrameGrayscaleByteSecondaryCaptureImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
    generate_uid,
)

# Fixed project root used by segAuto inference.
MAINDIR = Path("/home/zchen/Code/NeUF/code_seg_3D")


def _parse_mhd(mhd_path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    for raw_line in mhd_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def _resolve_paths(datapath: Path) -> Tuple[Path, Path]:
    if datapath.is_dir():
        mhd_files = sorted(datapath.glob("*.mhd"))
        if len(mhd_files) != 1:
            raise ValueError(
                f"Expected exactly one .mhd in directory {datapath}, found {len(mhd_files)}"
            )
        mhd_path = mhd_files[0]
    elif datapath.suffix.lower() == ".mhd":
        mhd_path = datapath
    elif datapath.suffix.lower() == ".raw":
        mhd_path = datapath.with_suffix(".mhd")
        if not mhd_path.exists():
            raise FileNotFoundError(f"Missing paired mhd file: {mhd_path}")
    else:
        raise ValueError("datapath must be a directory, .mhd file, or .raw file")

    mhd_info = _parse_mhd(mhd_path)
    raw_name = mhd_info.get("ElementDataFile")
    if not raw_name:
        raise ValueError(f"ElementDataFile not found in {mhd_path}")

    raw_path = (mhd_path.parent / raw_name).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    return mhd_path, raw_path


def _dtype_from_mhd(element_type: str) -> np.dtype:
    m = {
        "MET_UCHAR": np.uint8,
        "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_UINT": np.uint32,
        "MET_INT": np.int32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    try:
        return np.dtype(m[element_type])
    except KeyError as exc:
        raise ValueError(f"Unsupported ElementType: {element_type}") from exc


def _load_volume_from_mhd_raw(mhd_path: Path, raw_path: Path) -> np.ndarray:
    mhd_info = _parse_mhd(mhd_path)

    dim_tokens = mhd_info.get("DimSize", "").split()
    if len(dim_tokens) != 3:
        raise ValueError(f"Invalid DimSize in {mhd_path}: {mhd_info.get('DimSize')}")
    x, y, z = [int(v) for v in dim_tokens]

    channels = int(mhd_info.get("ElementNumberOfChannels", "1"))
    if channels != 1:
        raise ValueError(f"Only single-channel volume supported, got {channels}")

    element_type = mhd_info.get("ElementType")
    if not element_type:
        raise ValueError(f"ElementType not found in {mhd_path}")

    dtype = _dtype_from_mhd(element_type)
    arr = np.fromfile(raw_path, dtype=dtype)
    expected = x * y * z
    if arr.size != expected:
        raise ValueError(
            f"RAW size mismatch: expected {expected} voxels from DimSize={x} {y} {z}, got {arr.size}"
        )

    # MHD DimSize is X Y Z, and RAW is stored with X as fastest axis.
    vol_zyx = arr.reshape((z, y, x))
    return vol_zyx


def _to_uint8(vol: np.ndarray) -> np.ndarray:
    if vol.dtype == np.uint8:
        return vol

    vol = vol.astype(np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.uint8)
    scaled = (vol - vmin) / (vmax - vmin)
    return (scaled * 255.0).round().astype(np.uint8)


def _write_multiframe_dicom(volume_zyx_u8: np.ndarray, output_path: Path, ref: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z, y, x = volume_zyx_u8.shape

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MultiFrameGrayscaleByteSecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    now = dt.datetime.now()

    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"

    ds.PatientName = ref
    ds.PatientID = ref

    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesDate = ds.StudyDate
    ds.SeriesTime = ds.StudyTime

    ds.Rows = y
    ds.Columns = x
    ds.NumberOfFrames = str(z)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    ds.PixelSpacing = ["1", "1"]
    ds.SliceThickness = "1"

    ds.PixelData = np.ascontiguousarray(volume_zyx_u8).tobytes()

    ds.save_as(str(output_path), write_like_original=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw+mhd to DICOM and run segAuto inference"
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Case ID used for directory and file naming (e.g. ckpt_2)",
    )
    parser.add_argument(
        "--datapath",
        required=True,
        help="Path to input directory/.mhd/.raw",
    )
    args = parser.parse_args()

    datapath = Path(args.datapath).expanduser().resolve()
    if not datapath.exists():
        raise FileNotFoundError(f"datapath not found: {datapath}")

    mhd_path, raw_path = _resolve_paths(datapath)
    vol_zyx = _load_volume_from_mhd_raw(mhd_path, raw_path)
    vol_u8 = _to_uint8(vol_zyx)

    repere_dir = MAINDIR / "Pre_traitement_echo_v2" / "Repere_commun" / args.ref
    dcm_path = repere_dir / f"data_repcom_{args.ref}.dcm"

    _write_multiframe_dicom(vol_u8, dcm_path, args.ref)
    print(f"[OK] DICOM written: {dcm_path}")

    # Make local imports work without requiring external PYTHONPATH export.
    if str(MAINDIR) not in sys.path:
        sys.path.insert(0, str(MAINDIR))

    from lib.segAuto.src.inference.run_inference import run

    run(dataset_path=str(repere_dir), maindir=str(MAINDIR), ref=args.ref)


if __name__ == "__main__":
    main()
