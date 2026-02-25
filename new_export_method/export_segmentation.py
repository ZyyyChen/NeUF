# export_segmentation.py
# This code aims to export 3D meshes using the segmentation's output as input using the marching cubes algorithm.

import os
import numpy as np
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import cv2

from volume_data import VolumeData, get_frames

def export_mesh_from_volume(
    volume,
    volume_data,
    step_size=3,
    mesh_out=None,
    base_color=(0.9, 0.2, 0.2, 1.0),
    metallic=0.0,
    roughness=0.9,
):

    verts, faces, normals, _ = marching_cubes(volume=volume,
                                                   step_size=step_size,
                                                   spacing=volume_data.spacing
                                                   )

    # show_mesh(verts, faces)

    # save the surface as glb file
    mesh = trimesh.Trimesh(verts, faces, normals)

    print("volume_data.origin:", volume_data.origin)
    mesh.apply_translation(volume_data.origin)

    mesh = add_visual_to_mesh(mesh, base_color=base_color, metallic=metallic, roughness=roughness)

    if mesh_out is None:
        mesh_out = Path("c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/isosurface.glb")
    else:
        mesh_out = Path(mesh_out)
    mesh_out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(mesh_out), file_type='glb', include_normals=True)
    return str(mesh_out)


def get_mesh_from_frames(
    seg_frames,
    volume_data,
    step_size=3
    ):
    volume = np.stack(seg_frames, axis=0)
    
    verts, faces, normals, _ = marching_cubes(volume=volume,
                                                   step_size=step_size,
                                                   spacing=volume_data.spacing
                                                   )
    mesh = trimesh.Trimesh(verts, faces, normals)
    mesh.apply_translation(volume_data.origin)

    return mesh


def export_bbox_mesh(
    bounding_box,
    for_viewer=False,
    base_color=(0.1, 0.6, 0.9, 1.0),
    metallic=0.0,
    roughness=0.9,
):
    min_point, max_point = bounding_box[0], bounding_box[1]
    corners = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        # [1, 2, 6], [1, 6, 5],  # Right
        # [0, 3, 7], [0, 7, 4],  # Left
    ])
    mesh = trimesh.Trimesh(vertices=corners, faces=faces)

    mesh = add_visual_to_mesh(mesh, base_color=base_color, metallic=metallic, roughness=roughness)

    if for_viewer:
        mesh_out = Path("c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/target.glb")
    else:
        mesh_out = Path("c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/bbox.glb")
    mesh.export(str(mesh_out), file_type='glb', include_normals=True)


def show_mesh(verts, faces):
    
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')

    # plot surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    plt.tight_layout()
    plt.show()


def add_visual_to_mesh(mesh, base_color=(0.9, 0.2, 0.2, 1.0), metallic=0.0, roughness=0.9):
    # Apply material / color: prefer PBR material, fallback to solid vertex color
    try:
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
        material = PBRMaterial(
            name="Segmentation",
            baseColorFactor=base_color,
            metallicFactor=metallic,
            roughnessFactor=roughness,
        )
        mesh.visual = TextureVisuals(uv=None, image=None, material=material)
    except Exception:
        # Fallback to per-vertex solid color (RGBA 0-255)
        rgba = np.clip(np.array(base_color) * 255.0, 0, 255).astype(np.uint8)
        if rgba.shape[0] == 4:
            colors = np.tile(rgba, (len(mesh.vertices), 1))
            mesh.visual.vertex_colors = colors
    return mesh


def get_color_values(color_name):
    color_dict = {
        "red": (0.9, 0.2, 0.2, 1.0),
        "green": (0.2, 0.9, 0.2, 1.0),
        "blue": (0.2, 0.2, 0.9, 1.0),
        "yellow": (0.9, 0.9, 0.2, 1.0),
        "cyan": (0.2, 0.9, 0.9, 1.0),
        "magenta": (0.9, 0.2, 0.9, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5, 1.0),
        "black": (0.0, 0.0, 0.0, 1.0),
    }
    return color_dict.get(color_name.lower(), (0.9, 0.2, 0.2, 1.0))


def export_scene_from_meshes(meshes, export_dir, label="target"):
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        # Ensure each material has a unique name so exporters don't merge them
        try:
            if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                mesh.visual.material.name = f"Segmentation_{label}_{i+1}"
        except Exception:
            pass
        scene.add_geometry(mesh, node_name=f"seed_{label}_{i+1}")

    if len(meshes) == 0: # empty scene fill with a tiny box
        box = trimesh.creation.box(extents=(0.1,0.1,0.1))
        box.apply_translation((0,0,0))
        box = add_visual_to_mesh(box, base_color=get_color_values("red"))
        scene.add_geometry(box, node_name="empty")

    scene.export(os.path.join(export_dir, label + ".glb"), file_type='glb')


def get_volume_centroids(volume):
    centroids = []
    for i, frame in enumerate(volume):
        centroid = None
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
        centroids.append(centroid)

    return centroids


def get_centroid_points(volume, spacing, origin):
    centroids = get_volume_centroids(volume)
    points = []
    for i, centroid in enumerate(centroids):
        if centroid is not None:
            cx, cy = centroid
            # Convert pixel coordinates to physical space
            x = i * spacing[0] + origin[0]    # spacing[0] = slice thickness
            y = cy * spacing[2] + origin[1]   # spacing[2] = pixel size in y
            z = cx * spacing[1] + origin[2]   # spacing[1] = pixel size in x
            points.append([x, y, z])
    return np.array(points)

def smooth_points(points, nb_of_point=3):
    smoothed_points = []
    for i in range(len(points)):
        start_idx = max(0, i - nb_of_point // 2)
        end_idx = min(len(points), i + nb_of_point // 2 + 1)
        neighborhood = points[start_idx:end_idx]
        smoothed_point = np.mean(neighborhood, axis=0)
        smoothed_points.append(smoothed_point)

    return np.array(smoothed_points)

def centroids_to_line_mesh(points):
    # Create edges between consecutive points
    smoothed_points = smooth_points(points, nb_of_point=5)
    edges = [[i, i+1] for i in range(len(points)-1)]
    path = trimesh.load_path(smoothed_points[edges])
    return path


def get_skeleton_from_frames(seg_frames, volume_data):
    volume = np.stack(seg_frames, axis=0)
    points = get_centroid_points(volume, volume_data.spacing, volume_data.origin)
    line_mesh = centroids_to_line_mesh(points)
    return line_mesh


def run_export_segmentation(
        seg_dir, 
        volume_data_path, 
        mesh_out,
        phantom_type="bluephantom", 
        step_size=1) -> str:
    seg_dir = Path(seg_dir)
    mesh_out = Path(mesh_out)
    
    export_dir = Path(str(seg_dir).split("segmentation")[0])
    
    volume_data_path = Path(volume_data_path)
    volume_data = VolumeData.load(str(volume_data_path))

    seed_list = [f for f in os.listdir(seg_dir)  if os.path.isdir(os.path.join(seg_dir,f))]
    
    meshes = []
    for i,seed_name in enumerate(seed_list):
        print("Working on seed:", i)
        seg_frames = get_frames(os.path.join(seg_dir, seed_name))
        mesh = get_mesh_from_frames(seg_frames=seg_frames,
                                    volume_data=volume_data,
                                    step_size=step_size,
                                    )
        meshes.append(mesh)
        mesh.export(str(Path(seg_dir)/f"seed_{i}_isosurface.glb"), file_type='glb', include_normals=True)
        # add laplacien smoothing
        # mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=5)
        # mesh.export(str(Path(seg_dir)/f"seed_{i}_isosurface_smoothed.glb"), file_type='glb', include_normals=True)
        
    # phantom type
    if phantom_type == "bluephantom": # only target(s)
        labels = ["target" for _ in range(len(meshes))]
    elif phantom_type == "simu":  # only target(s)
        labels = ["target" for _ in range(len(meshes))]
    elif phantom_type == "jfr": # target + multiple ribs
        labels = ["target"] + ["obstacle" for _ in range(len(meshes)-1)] 
    else:
        raise RuntimeError(f"Phantom type {phantom_type} undefined")
    
    target_meshes, obstacle_meshes = [], []
    for i,mesh in enumerate(meshes):
        label = labels[i]
        if label == "target":
            target_meshes.append(
                add_visual_to_mesh(
                    mesh, base_color=get_color_values("yellow")))
        else:
            obstacle_meshes.append(
                add_visual_to_mesh(
                    mesh, base_color=get_color_values("white")))

    export_scene_from_meshes(target_meshes,export_dir,"target")
    export_scene_from_meshes(obstacle_meshes,export_dir,"ribs")
    
    return seg_dir


if __name__ == "__main__":
    folder = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/"
    seg_folder = os.path.join(folder, "segmentation")
    reprod_folder = os.path.join(folder, "reprod")

    seed_folder_list = [f for f in os.listdir(seg_folder)  if os.path.isdir(os.path.join(seg_folder,f))]

    # Load VolumeData
    volume_data_path = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/volume_data.json"
    volume_data = VolumeData.load(volume_data_path)
    
    if len(seed_folder_list)==0:
        seg_frames = get_frames(seg_folder, reversed=False)
        volume = np.stack(seg_frames, axis=0)
        print(volume.shape)


        export_mesh_from_volume(volume=volume,
                                    volume_data=volume_data,
                                    step_size=1)
    
    else:
        ribs_meshes, skeletons = [], [] 
        colors = [(1.0,1.0,1.0, 1.0), (0.2, 0.9, 0.2, 1.0), (1.0, 1.0, 0.0, 1.0)]
        for i,seed_folder in enumerate(seed_folder_list):
            print("Working on seed:", i)
            seg_frames = get_frames(os.path.join(seg_folder, seed_folder))
            volume = np.stack(seg_frames, axis=0)
            skeleton = get_skeleton_from_frames(seg_frames, volume_data)
            verts, faces, normals, _ = marching_cubes(volume=volume,
                                                step_size=1,
                                                spacing=volume_data.spacing
                                                )
            mesh = trimesh.Trimesh(verts, faces, normals)
            mesh.apply_translation(volume_data.origin)
            skeleton.apply_translation(volume_data.origin)
            if i == 0: # seed nb 0 = target 
                target = add_visual_to_mesh(mesh, base_color=colors[2], metallic=0.0, roughness=0.9)    
            else: # obstacles ribs
                mesh = add_visual_to_mesh(mesh, base_color=colors[0], metallic=0.0, roughness=0.9)
                ribs_meshes.append(mesh)
                skeletons.append(skeleton)
            mesh.export(str(Path(seg_folder)/f"seed_{i}_isosurface.glb"), file_type='glb', include_normals=True)
        # Export a single GLB containing multiple geometries, each with its own material
        scene = trimesh.Scene()
        for i, mesh in enumerate(ribs_meshes):
            # Ensure each material has a unique name so exporters don't merge them
            try:
                if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                    mesh.visual.material.name = f"Segmentation_{i+1}"
            except Exception:
                pass
            scene.add_geometry(mesh, node_name=f"seed_{i+1}")
            scene.add_geometry(skeletons[i], node_name=f"seed_skeleton_{i+1}")

        isosurface_mesh_out = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/target.glb"
        target.export(str(isosurface_mesh_out), file_type='glb')

        ribs_mesh_out = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/latest/sync/export/ribs.glb"
        scene.export(str(ribs_mesh_out), file_type='glb')

    export_bbox_mesh(bounding_box= [volume_data.point_min, volume_data.point_max], 
                for_viewer=False)
    
    # reprod folder --> isosurface.glb
    seg_frames = get_frames(reprod_folder, reversed=False)
    volume = np.stack(seg_frames, axis=0)
    print(volume.shape)

    export_mesh_from_volume(volume=volume,
                                volume_data=volume_data,
                                step_size=1)