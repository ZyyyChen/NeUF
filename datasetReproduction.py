from dataset import Dataset
import matplotlib.pyplot as plt
from slice_renderer import SliceRenderer
from nerf_network import NeRF
from utils import *
import os
from tqdm import tqdm
import cv2

def getRendererfromCkpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    nerf = NeRF(ckpt)
    dataset = Dataset.open_from_save(ckpt["baked_dataset_file"])
    slice_renderer = SliceRenderer(dataset)

    return nerf, slice_renderer

def slicesRendering(ckpt_path,output_path, number_of_slices=15, scale=None,crop_info=None):
    nerf, slice_renderer = getRendererfromCkpt(ckpt_path)

    out_folder = f"{output_path}/reprod/"

    os.makedirs(out_folder, exist_ok=True)

    if crop_info:
        start_w, start_h, size_w, size_h = crop_info
    else:
        start_w, start_h, size_w, size_h = 0, 0, slice_renderer.width_px, slice_renderer.height_px

    for i in tqdm(range(0,number_of_slices),total=number_of_slices):
        img = slice_renderer.render_slice_from_dataset(nerf,i,reshaped=True,scalefactor=scale).detach().cpu().numpy()
        img = img[start_h*scale:(start_h+size_h)*scale,start_w*scale:(start_w+size_w)*scale]
        # plt.imsave(f"{out_folder}/{str(i)}.jpg",img,cmap="gray")
        # save the image as a one channel image
        cv2.imwrite(f"{out_folder}/us{str(i)}.jpg", img)

def oneSliceRendering(ckpt_path, num_slice, output_dir, suffix, scale=None,crop_info=None):
    nerf, slice_renderer = getRendererfromCkpt(ckpt_path)

    os.makedirs(output_dir, exist_ok=True)

    img = slice_renderer.render_slice_from_dataset(model=nerf,
                                                   slice_number=num_slice,
                                                   reshaped=True,
                                                   scalefactor=scale
                                                   ).detach().cpu().numpy()

    plt.imsave(f"{output_dir}/{suffix}.png",img,cmap="gray")

def getOneSliceFromRenderer(ckpt_path, num_slice):
    nerf, slice_renderer = getRendererfromCkpt(ckpt_path)
    
    return slice_renderer.render_slice_from_dataset(model=nerf,
                                                   slice_number=num_slice,
                                                   reshaped=True,
                                                   scalefactor=2
                                                   ).detach().cpu().numpy()

if __name__ == "__main__":
    num_slice = 200
    for image_step in [10]:#, 2, 5, 10]:
        for num_scan in ["01"]:#,"03","07","11"]:
            root_folder = f"c:/Users/anchling/Documents/projects/neural-ultrasound-field/logs/03-06-2025/save_every_5000/image_step={image_step}"
            ckpts_folder = os.path.join(root_folder,f"scan_{num_scan}","checkpoints")

            ckpts = os.listdir(ckpts_folder)
            for ckpt_path in tqdm(ckpts, total=len(ckpts),desc=f"Working on scan {num_scan}"):
                oneSliceRendering(os.path.join(ckpts_folder,ckpt_path), 
                                num_slice=int(num_slice/image_step),
                                output_dir=os.path.join(root_folder,"evolution",f"scan_{num_scan}",f"slice_{num_slice}"),
                                suffix=os.path.basename(ckpt_path).split(".")[0],
                                scale=1, 
                                crop_info=None)