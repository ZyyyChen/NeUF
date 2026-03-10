from slice_renderer import SliceRenderer
from nerf_network import NeRF
from dataset_1 import Quat, Dataset
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import scipy.io
import os

def clamp(val):
    return 0 if val < 0 else 255 if val > 255 else val

def model_to_raw(ckpt, destination_file, point_min, point_max, decim):
    values = []

    nerf = NeRF(ckpt)

    width = ceil(point_max[0] - point_min[0])
    height = ceil(point_max[1] - point_min[1])
    depth = ceil(point_max[2] - point_min[2])
    px_width = int(width / decim)
    px_height = int(height / decim)
    px_depth = int(depth / decim)

    K = np.linspace(1 / (2 * px_depth), 1 - 1 / (2 * px_depth), px_depth)
    J = np.linspace(1 / (2 * px_height), 1 - 1 / (2 * px_height), px_height)
    I = np.linspace(1 / (2 * px_width), 1 - 1 / (2 * px_width), px_width)

    print("dim : ", px_width, px_height, px_depth)

    # render = SliceRenderer(px_width=px_width,
    #                        px_height=px_height,
    #                        width=width,
    #                        height=height,
    #                        point_min=point_min,
    #                        point_max=point_max,
    #                        decimation=decim)



    print("generating positions")
    m = np.meshgrid(J * height + point_min[1], K * depth + point_min[2], I * width + point_min[0])
    m = m[2], m[0], m[1]

    points = np.zeros((px_height * px_depth * px_width, 3))
    for i, arr in enumerate(m):
        points[:, i] = arr.flatten()

    positions = torch.FloatTensor(points).to("cuda")

    # positions = []

    # for k in np.linspace(1/(2*px_depth),1-1/(2*px_depth),px_depth):
    #     for j in np.linspace(1/(2*px_height),1-1/(2*px_height),px_height):
    #         for i in np.linspace(1/(2*px_width),1-1/(2*px_width),px_width):
    #             positions.append(torch.FloatTensor([i*width+point_min[0],
    #                                                 j*height+point_min[1],
    #                                                 k*depth+point_min[2]]
    #                                                ).to("cuda"))
    #
    # # print(positions)
    # positions = torch.stack(positions)
    # #
    i = 0
    with open("debugpy.txt","w") as f :
        for p in positions.detach().cpu().numpy() :
            f.write(str(i) + " " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
            i += 1

    print("evaluating network")
    if nerf.encoding_type != "HASH" or not nerf.use_encoding:
        positions = torch.add(
            torch.multiply(torch.divide(torch.add(positions, -torch.FloatTensor(point_min).to("cuda")), torch.max(torch.FloatTensor(
            (point_max[0] - point_min[0], point_max[1] - point_min[1], point_max[2] - point_min[2]))).to(
            "cuda")), 2), -1)

    #todo handle directions
    with torch.no_grad():
        data = nerf.query(positions,positions).detach().cpu().numpy()


    for d in data :
        values.append(int(clamp(d)).to_bytes(length=1, byteorder="big"))

    with open(destination_file, "wb") as f:
        for val in values:
            f.write(val)

    print("PARAVIEW :")
    print("\t 0 -", px_width-1)
    print("\t 0 -", px_height-1)
    print("\t 0 -", px_depth-1)

    return (px_width-1, px_height-1, px_depth-1)

def model_to_raw_old(ckpt, destination_file, point_min, point_max, decim):
    values = []

    # nerf = NeRF(ckpt)
    #
    # width = ceil(point_max[0] - point_min[0])
    # height = ceil(point_max[1] - point_min[1])
    # depth = ceil(point_max[2] - point_min[2])
    # px_width = int(width / decim)
    # px_height = int(height / decim)
    # px_depth = int(depth / decim)

    #
    width = 10
    height = 15
    depth = 20
    px_width = 10
    px_height = 15
    px_depth = 20

    print("dim : ", px_width, px_height, px_depth)

    # render = SliceRenderer(px_width=px_width,
    #                        px_height=px_height,
    #                        width=width,
    #                        height=height,
    #                        point_min=point_min,
    #                        point_max=point_max,
    #                        decimation=decim)

    data = None
    nbslice = 0

    totalValue = 255
    print(totalValue)

    for n, i in tqdm.tqdm(enumerate(np.linspace(0, 20, px_depth))):
    # for n, i in tqdm.tqdm(enumerate(np.linspace(point_max[2], point_min[2], px_depth))):
    # for n, i in enumerate(np.linspace(point_min[2], point_max[2], px_depth)):
        data = np.reshape(np.linspace(i*(totalValue/px_depth),(i+1)* (totalValue/px_depth),px_width*px_height)
                          ,(px_height,px_width))

    #     data = np.array(render.render_slice(nerf,
    #                                         [(point_min[0] + point_max[0])/2, point_min[1] , i],
    #                                         Quat(0.5,-0.5,-0.5,-0.5),
    #                                         ).detach().cpu())
        plt.imsave("../volume/test/" + str(n) + ".jpg", data)
        nbslice += 1

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                values.append(int(clamp(data[x][y])).to_bytes(length=1, byteorder="big"))

    with open(destination_file, "wb") as f:
        for val in values:
            f.write(val)

    print("PARAVIEW :")
    print("\t 0 -", data.shape[1] - 1)
    print("\t 0 -", data.shape[0] - 1)
    print("\t 0 -", nbslice - 1)

    return (data.shape[1] - 1, data.shape[0] - 1, nbslice - 1)



def export_volume_series(source,folder,step,number):
    ckpt = torch.load(source)
    if "bounding_box" in ckpt.keys():
        point_min, point_max = ckpt["bounding_box"]
    else:
        point_min, point_max = ckpt["bouding_box"]


    x, y, z = model_to_raw(ckpt, folder + "/volume"+str(number)+".raw", point_min.detach().cpu().numpy(),
                               point_max.detach().cpu().numpy(), step)

    create_info_file(x,y,z,folder,source,step)


def export_volume(source,dest,dest_folder,step) :
    ckpt = torch.load(source, weights_only=False)
    if "bounding_box" in ckpt.keys() :
        point_min, point_max = ckpt["bounding_box"]
    else :
        point_min, point_max = ckpt["bouding_box"]

    x,y,z = model_to_raw(ckpt,dest_folder+"/volume.raw",point_min.detach().cpu().numpy(),point_max.detach().cpu().numpy(),step)
    # x,y,z = model_to_raw(None,dest_folder+"/volume.raw",None,None,1)

    create_parafile(x,y,z,dest,dest_folder)
    create_info_file(x,y,z,dest_folder,source,step)


def create_info_file(x,y,z,dest_folder,source,step) :
    with open(dest_folder+"/info.txt","w") as f:
        f.write("PARAVIEW DIMENSIONS :\n")
        f.write("\t 0 - " + str(x) + "\n")
        f.write("\t 0 - " + str(y) + "\n")
        f.write("\t 0 - " + str(z) + "\n")
        f.write("\nDiscretization: " + str(step) + "mm\n")
        f.write("\nSource:\n")
        f.write(source)
        f.write("\n")


def create_parafile(x,y,z,dest,dest_folder):
    param_path = os.path.join(os.path.dirname(__file__),"interface","base_para.txt")
    with open(param_path,"r") as f:
        text = f.read()

    text = text.format(X=x,Y=y,Z=z,dest=dest)

    with open(dest_folder+"/para.py","w") as fi :
        fi.write(text)

if __name__ == "__main__" :
    x,y,z = model_to_raw(None,"volume.raw",None,None,1)