import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def average_rotation(infos_file_path="c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/test_brasGot_50x50/sync/export/infos.dat"):
    with open(infos_file_path, "r") as info:
        lines = info.readlines()[1:]
    quaternions = np.array([list(map(float,l.split("\n")[0].split(" ")[-6:-2])) for l in lines])
    quaternions[:,1:] *= -1
    # Average rotation: convert quaternions to rotation matrices
    rot_mats = R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix() # convert (w,x,y,z) → (x,y,z,w)
    avg_rot = np.mean(rot_mats, axis=0)
    # avg_rot = np.mean(rot_mats, axis=0)
    # # Re-orthogonalize (SVD projection)
    U, _, Vt = np.linalg.svd(avg_rot)
    R_mat = U @ Vt

    # Assemble 4x4 matrix
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = avg_rot

    return R_4x4

def to_vtk_transform(matrix):
    """ Convert a numpy matrix to a VTK transform."""
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix[i, j])
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    return transform

def build_combined_matrix_6dof(infos):
    """Build a transformation matrix that matches the dataset generation process."""
    px_width, px_height = 50, 50  # Default values from dataset
    dim_x, dim_y, dim_z = infos["dimensions"]
    offset = infos["offset"]
    
    # 1. Initial centering (matching get_base_points in utils.py)
    pixel_size_w = dim_x / px_width
    pixel_size_h = dim_y / px_height
    
    # Center width around 0, height from 0 (matching dataset generation)
    T_center = np.eye(4)
    T_center[:3, 3] = [-dim_x/2 + pixel_size_w/2,  # Center X (width)
                       0 + pixel_size_h/2,          # Start Y from 0 (height)
                       -dim_z/2 + pixel_size_w/2]   # Center Z (depth)
    
    # 2. Scale to match pixel dimensions
    S = np.diag([dim_x/px_width,      # X scale
                 dim_y/px_height,      # Y scale 
                 dim_z/px_height,      # Z scale
                 1])
    
    # 3. Rotation to match ultrasound probe orientation
    # This matches the orientation used in get_oriented_points_and_views
    Rx90 = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    
    # 4. Translation to final position
    T = np.eye(4)
    T[:3, 3] = offset
    
    # Combine transforms in the correct order
    # The order matches how points are transformed in the dataset:
    # 1. Center and scale the base points
    # 2. Apply rotation
    # 3. Apply position offset
    M = T @ Rx90 @ S @ T_center
    
    return M

def build_combined_matrix_6dof_GE(infos):
    """Build a transformation matrix that matches the dataset generation process with the new GE US Probe sensor."""
    px_width, px_height = 45, 30  # Default values from dataset
    dim_x, dim_y, dim_z = infos["dimensions"]
    offset = infos["offset"]
    
    # 1. Initial centering (matching get_base_points in utils.py)
    pixel_size_w = dim_x / px_width
    pixel_size_h = dim_y / px_height
    
    # Center width around 0, height from 0 (matching dataset generation)
    T_center = np.eye(4)
    T_center[:3, 3] = [-dim_x/2 + pixel_size_w/2,  # Center X (width)
                       0 + pixel_size_h/2,          # Start Y from 0 (height)
                       -dim_z/2 + pixel_size_w/2]   # Center Z (depth)
    
    # 2. Scale to match pixel dimensions
    S = np.diag([dim_x/px_width,      # X scale
                 dim_y/px_height,      # Y scale 
                 dim_z/px_height,      # Z scale
                 1])
    print(S)
    # 3. Rotation to match ultrasound probe orientation
    # This matches the orientation used in get_oriented_points_and_views
    Rx90 = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    Rxm90 = np.array([
        [1, 0,  0, 0],
        [0, 0, 1, 0],
        [0, -1,  0, 0],
        [0, 0,  0, 1]
    ])
    
    # 4. Translation to final position
    T = np.eye(4)
    T[:3, 3] = offset
    
    # Combine transforms in the correct order
    # The order matches how points are transformed in the dataset:
    # 1. Center and scale the base points
    # 2. Apply rotation
    # 3. Apply position offset
    M = T @ Rxm90
    # M = np.eye(4)
    return M

def build_combined_matrix(dimensions, orig_dims, roi=None):
    """Build a transformation matrix for ROI-based transformations."""
    dim_x, dim_y, dim_z = dimensions
    
    # Base transform (same as build_combined_matrix_6dof)
    base_transform = build_combined_matrix_6dof({
        "dimensions": dimensions,
        "offset": [0, 0, 0],  # No offset for base transform
    })
    
    # Additional ROI transform if needed
    if roi:
        # Scale for ROI
        scale_h = roi['height'] / orig_dims[1]
        scale_w = roi['width'] / orig_dims[0]
        S_ROI = np.diag([scale_h, scale_w, 1, 1])
        
        # ROI translation
        T_ROI = np.eye(4)
        T_ROI[:3, 3] = [
            roi['x'] / orig_dims[0] * dimensions[0],  # X translation
            roi['y'] / orig_dims[1] * dimensions[1],  # Y translation
            0                                         # No Z translation
        ]
        
        return T_ROI @ S_ROI @ base_transform
    
    return base_transform

def getSimpleTransform(infos):
    """Get the complete transform that matches dataset generation."""
    matrix = build_combined_matrix_6dof_GE(infos)
    return to_vtk_transform(matrix)

def getSimpleTransformOld(dimensions, orig_dims, roi=None):
    """
    Create a transformation without full 6DOF pose, including:
    - Initial volume reorientation and centering
    - Rotation fix (180° X)
    - ROI scaling and translation
    """

    # 1. Reorient and center volume in scanner coordinate frame
    reorient_transform = vtk.vtkTransform()
    reorient_transform.RotateX(90)
    reorient_transform.Translate(-dimensions[0] / 2, 0, -dimensions[2] / 2)
    reorient_transform.Scale(dimensions[2] / dimensions[0], 1, 1)  # X-scaling

    # 2. Flip to match acquisition orientation (commonly needed)
    fix_rotation_transform = vtk.vtkTransform()
    fix_rotation_transform.RotateX(180)

    # 3. Translate to align volume to expected origin (e.g., scanner space)
    alignment_transform = vtk.vtkTransform()
    alignment_transform.Translate(0, 0, dimensions[1])

    # 4. Crop and scale for ROI
    crop_scale_transform = vtk.vtkTransform()
    crop_translate_transform = vtk.vtkTransform()
    if roi:
        scale_h = roi['height'] / orig_dims[1]
        scale_w = roi['width'] / orig_dims[0]
        crop_scale_transform.Scale(scale_h, scale_w, 1)

        # Translate center to cropped center
        center_shift_x = (-1 + scale_w) * (dimensions[0] / 2)
        center_shift_y = (-1 + scale_h) * (dimensions[2] / 2)
        crop_translate_transform.Translate(center_shift_y, center_shift_x, 0)

        # Translate to correct ROI origin
        delta_y = roi['x'] / orig_dims[0] * dimensions[0]
        delta_x = roi['y'] / orig_dims[1] * dimensions[2]
        crop_translate_transform.Translate(delta_x, delta_y, 0)

    # Compose all transforms
    final_transform = vtk.vtkTransform()
    final_transform.Concatenate(crop_translate_transform)
    final_transform.Concatenate(crop_scale_transform)
    final_transform.Concatenate(alignment_transform)
    final_transform.Concatenate(fix_rotation_transform)
    final_transform.Concatenate(reorient_transform)

    print(final_transform.GetMatrix())

    return final_transform

def paraIsoSurface(raw_path: str,
                    out_folder: str,
                    dimensions: list,
                    offset: list,
                    center: list,
                    orig_dims: tuple,
                    roi: dict,
                    threshold: float = 127.5,
                    infos_file_path: str = None) -> None:
    """Function that creates an isosurface from a raw image using VTK and saves it as an OBJ file."""

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1216, 794)

    # Create a render window interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Create a camera and set its properties
    camera = vtk.vtkCamera()
    camera.SetPosition(97.12172294334181, 200.73455602133814, 99.10075349074762)
    camera.SetFocalPoint(30.000000000000007, 30.499999999999968, 27.499999999999975)
    camera.SetViewUp(-0.17947903253402203, 0.4406975393908316, -0.8795299629094611)
    renderer.SetActiveCamera(camera)

    # Read the raw image data
    reader = vtk.vtkImageReader()
    reader.SetDataScalarTypeToUnsignedChar()
    reader.SetDataExtent(0, dimensions[0], 0, dimensions[1], 0, dimensions[2])
    reader.SetFileDimensionality(3)
    reader.SetFileName(raw_path)
    reader.SetDataByteOrderToBigEndian()  # Match the byte order used when writing the file

    # Create a volume mapper
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    # Create a volume property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # Create a piecewise function for opacity transfer
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(11.0, 0.0)
    opacityTransferFunction.AddPoint(153.71697998046875, 0.0)
    opacityTransferFunction.AddPoint(255.0, 1.0)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    # Create a volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Add the volume to the renderer
    renderer.AddVolume(volume)
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Create a contour filter to extract isosurfaces
    contourFilter = vtk.vtkContourFilter()
    contourFilter.SetInputConnection(reader.GetOutputPort())
    contourFilter.SetValue(threshold, threshold)  # Set the isovalue for the contour

    # Create a vtkTransform for centering
    center_transform = vtk.vtkTransform()
    center_transform.Translate(-center[0], -center[1], -center[2])
    
    infos = {
        "dimensions":[dimensions[0]+1,
                      dimensions[1]+1,
                      dimensions[2]+1],
        "offset":offset,
        "center":center,
        "orig_dims":orig_dims,
        "roi":roi,
    }
    final_transform = getSimpleTransform(infos)

    # Apply the transform to the contour
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(contourFilter.GetOutputPort())
    transformFilter.SetTransform(final_transform)

    # Create a mapper for the contour
    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInputConnection(transformFilter.GetOutputPort())
    contourMapper.ScalarVisibilityOff()

    # Create an actor for the contour
    contourActor = vtk.vtkActor()
    contourActor.SetMapper(contourMapper)

    # Add the contour actor to the renderer
    renderer.AddActor(contourActor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Render interaction
    renderWindow.Render()

    # Export the contour in obj
    objExporter = vtk.vtkOBJExporter()
    objExporter.SetInput(renderWindow)
    objExporter.SetFilePrefix(out_folder + "/isosurface")
    objExporter.Write()

    # Start Interaction
    # renderWindowInteractor.Start()