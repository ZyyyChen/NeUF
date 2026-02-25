import bpy
import sys

folder_path = sys.argv[-1]

# Ensure the OBJ import add-on is enabled
# bpy.ops.preferences.addon_enable(module="io_scene_obj")

# Clear existing data
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the .obj file
bpy.ops.wm.obj_import(filepath=folder_path + "/isosurface.obj")

# Ensure materials are correctly assigned (if needed)
# You can customize this part to adjust materials as necessary

# Export to .glb format
bpy.ops.export_scene.gltf(filepath=folder_path + "/isosurface.glb", export_format='GLB')