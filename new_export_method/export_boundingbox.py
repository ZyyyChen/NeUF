from export_segmentation import export_bbox_mesh
import os, sys

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from dataset import Dataset

dataset = Dataset.open_from_save("c:/Users/anchling/Documents/projects/neural-ultrasound-field/datasets/baked/test/latest.pkl")

bounding_box = dataset.get_bounding_box()
point_min, point_max = bounding_box[0].detach().cpu().numpy(), bounding_box[1].detach().cpu().numpy()
bounding_box = (point_min, point_max)
print("bounding_box:", bounding_box)

export_bbox_mesh(bounding_box, for_viewer=True)
