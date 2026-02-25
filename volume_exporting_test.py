from nerf_network import NeRF
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, shutil
from datasetReproduction import slicesRendering, getOneSliceFromRenderer
import pyvista as pv
from paraviewIsoSurface import paraIsoSurface
import subprocess
from scripting_for_new_data.dicomGenerator import multi_frame_dicom_with_infosdat_positions
import argparse, json
from PIL import Image
import cv2

PATH = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/TEST_NeRFExport/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clamp(val):
    return 0 if val < 0 else 255 if val > 255 else val

def model_to_raw(ckpt, destination_file, point_min, point_max, decim, threshold):
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

    # dimensions = (px_depth, px_height, px_width)
    dimensions = (px_width, px_height, px_depth)
    print("dim : ", dimensions)

    # print("generating positions")
    m = np.meshgrid(J * height + point_min[1], K * depth + point_min[2], I * width + point_min[0])
    m = m[2], m[0], m[1]

    points = np.zeros((px_height * px_depth * px_width, 3))
    for i, arr in enumerate(m):
        points[:, i] = arr.flatten()

    positions = torch.FloatTensor(points).to("cuda")

    # print("evaluating network")
    if nerf.encoding_type != "HASH" or not nerf.use_encoding:
        positions = torch.add(
            torch.multiply(torch.divide(torch.add(positions, -torch.FloatTensor(point_min).to("cuda")), torch.max(torch.FloatTensor(
            (point_max[0] - point_min[0], point_max[1] - point_min[1], point_max[2] - point_min[2]))).to(
            "cuda")), 2), -1)

    #todo handle directions
    with torch.no_grad():
        data = nerf.query(positions,positions).detach().cpu().numpy()

    # plot_3d(points,positions.detach().cpu().numpy())
    print("data shape :", data.shape)

    # plot3DData(points, data_bis, threshold)

    # Ensure the data is in the correct shape
    volume_data = data.reshape(dimensions)

    # normalize the data
    nb_Bytes = 1
    normalized_data = normalize_data(data, nb_Bytes)
    
    plt.figure()
    plt.subplot(211)
    plt.hist(data, bins=150)
    plt.subplot(212)
    plt.hist(normalized_data, bins=150)
    # plt.show()
    # generate isosurface in vtk format
    # create_isoSurface(normalized_data, dimensions, threshold, PATH)
    
    # Convert the data to bytes
    for d in normalized_data.flatten():
        values.append(int(d).to_bytes(length=nb_Bytes, byteorder="big"))
    
    with open(destination_file, "wb") as f:
        for val in values:
            f.write(val)

    print("PARAVIEW :")
    print("\t 0 -", px_width-1)
    print("\t 0 -", px_height-1)
    print("\t 0 -", px_depth-1)

    return (px_width-1, px_height-1, px_depth-1)

def create_isoSurface(volume_data, dimensions, threshold, out_path="test_out"):
    """Generates vtk files for both the volume and the isosurface with a selected threshold"""
    # create volume
    grid = pv.ImageData(dimensions=dimensions)
    grid.point_data['values'] = volume_data.flatten(order='F')

    # create isosurface
    isosurface  = grid.contour([threshold])

    vtk_grid_path = out_path + '/volume.vtk'
    isoSurface_path = out_path + '/isosurface.vtk'
    grid.save(vtk_grid_path)
    isosurface.save(isoSurface_path)

def normalize_data(data, nb_Bytes=1):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val) * (2 ** (nb_Bytes * 8) - 1)

def plot3DData(points, data, threshold):
    values = data.flatten()

    # Thresholding to accelerate the ploting
    filtered_points = points[values > threshold]
    filtered_values = values[values > threshold]

    # 3D Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c=filtered_values, cmap='viridis')
    fig.colorbar(img)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_3d(data1, data2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2])
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2])
    ax.legend(['First','Second'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def export_volume(source,dest,dest_folder,step, threshold) :
    ckpt = torch.load(source, weights_only=False, map_location=device)
    print("dataset file : ",ckpt['baked_dataset_file'])
    print("Number of Iteration :", ckpt['start'])

    if "bounding_box" in ckpt.keys() :
        point_min, point_max = ckpt["bounding_box"]
    else :
        point_min, point_max = ckpt["bouding_box"]
    point_min, point_max = point_min.detach().cpu().numpy(), point_max.detach().cpu().numpy()

    dims2D = ckpt['dims2D'] if "dims2D" in ckpt.keys() else None
    roi2D = ckpt['roi2D'] if "roi2D" in ckpt.keys() else None

    # Get the dataset path from the checkpoint
    dataset_path = os.path.dirname(ckpt['baked_dataset_file'])
    infos_file_path = os.path.join(dataset_path, "sync/export/infos.dat")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    x,y,z = model_to_raw(ckpt,dest_folder+"/volume.raw",point_min,point_max,step,threshold)
    # x,y,z = model_to_raw(None,dest_folder+"/volume.raw",None,None,1)

    create_parafile(x,y,z,dest,dest_folder)
    create_info_file(x,y,z,dest_folder,source,step)

    return (x,y,z), point_min, point_max, dims2D, roi2D, infos_file_path

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

def onclick(event):
    # Check if the click is within the axes
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        intensity = reco_img[y, x]  # row = y, col = x
        # print(f"Clicked at (x={x}, y={y}) -> Intensity: {intensity}")
        threshold_info['x'] = x
        threshold_info['y'] = y
        threshold_info['intensity'] = intensity
        plt.close()

def getArgs():
    """To obtain the different arguments from the parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_folder', 
                        '-rf', 
                        type=str, 
                        help='Folder with input files (images and infos.dat file with positions)',
                        default= "latest",
                        # required=True
                        )
    return parser.parse_args()


if __name__ == "__main__" :
    file_dir = os.path.dirname(__file__)
    args = getArgs()
    
    ckpts = {
        # "bluePrintFab_works_old" : {"threshold": 100,
        #                 "step"  : 1},
        # "bluePrintFab_works_new" : {"threshold": 200,
        #                 "step"  : 1},
        # "bluePrintFabien_04_06_24_classic" : {"threshold": 100,
        #                 "step"  : 1},
        # "bluePrintFabien_04_06_24_rotated" : {"threshold": 150,
        #                 "step"  : 1},
        # "bluePrintFabien_28_05_24_classic" : {"threshold": 150,
        #                 "step"  : 1},
        # "bluePrintFabien_28_05_24_rotated" : {"threshold": 150,
        #                 "step"  : 1},   
        # "carotideFabien_26_06_24_classic" : {"threshold": 150,
        #                 "step"  : 1},
        # "carotideFabien_26_06_24_rotated" : {"threshold": 125,
        #                 "step"  : 1},   
        # "NewCarotideFabienFirst_21-03_25_classic" : {"threshold": 150,
        #                 "step"  : 1},  
        # "NewCarotideFabienFirst_21-03_25_rotated" : {"threshold": 120,
        #                 "step"  : 1},
        # "verylongscan10XandZslide" : {"threshold": 210,
        #                 "step"  : 1},
        # "scan_04_0.5spacing" : {"threshold": 243,
        #                 "step"  : 1},
        # "BPF_resized" : {"threshold": 180,
        #                 "step"  : 1},
        # "scan_03_new_pipeline" : {"threshold": 255-215,
        #                 "step"  : 1},
        # "scan_03_new_pipeline_crppped.pkl" : {"threshold": 215,
        #                 "step"  : 1},
        # "scan_01_full_pipeline_cropped_0.20_50x50" : {"threshold": 20 ,
        #                 "step"  : 1},
        # "scan_03_full_pipeline_cropped_0.20_50x50" : {"threshold": 125 ,
        #                 "step"  : 1},
        # "scan_07_full_pipeline_cropped_0.20_50x50" : {"threshold": 20 ,
        #                 "step"  : 1},
        # "scan_11_full_pipeline_cropped_0.20_50x50" : {"threshold": 70 ,
        #                 "step"  : 1},
        # "scan_01_90000" : {"threshold": 20 ,
        #                 "step"  : 1},
        # "scan_02_150000" : {"threshold": 70 ,
        #                 "step"  : 1},
        # "scan_03_250000" : {"threshold": 70 ,
        #                         "step"  : 1},
        "ckpt" : {"threshold":  50,
                        "step"  : 1},
        # "scan_03_cropped_skiin" : {"threshold":  20,
        #                  "step"  : 1},
        # "scan_01_fullsize" : {"threshold":  100,
        #                  "step"  : 1},
    }

    crop_infos = {
        "01" : (202, 4, 310, 178), # start_w, start_h, size_w, size_h
        "03" : (253, 12, 335, 213),
        "07" : (344, 31, 198, 192),
        "11" : (142, 183, 182, 171),
    }

    for key in ckpts.keys():
        ckpt_dict = ckpts[key]

        print("For {} training".format(key))
        ckpt_path = f"{file_dir}/latest/"+ key +".pkl"
        out_folder = f"{file_dir}/volume/Generated/tests/{key}"

        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)


        # Choice of threshold
        threshold_info = {}
        reco_img = getOneSliceFromRenderer(ckpt_path=ckpt_path,
                                           num_slice=20)
        # Display image
        fig, ax = plt.subplots()
        ax.imshow(reco_img, cmap='gray')
        ax.set_title("Click anywhere to get pixel intensity")

        # Connect the event
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        threshold = int(threshold_info["intensity"])
        # threshold = 50

        dimensions, point_min, point_max, orig_dims2D, roi2D, infos_file_path = export_volume(source = ckpt_path,
                    dest = "tests/{}".format(key),
                    dest_folder = out_folder,
                    step = ckpt_dict["step"],
                    threshold = threshold)
        center = (point_max - point_min) / 2

        paraIsoSurface(raw_path=out_folder+"/volume.raw", 
                       out_folder=PATH, 
                       dimensions=dimensions, 
                       offset=point_min,
                       center=center,
                       orig_dims=orig_dims2D,
                       roi = roi2D,
                       threshold=threshold,
                       infos_file_path=infos_file_path)
        
        # Construct the command to run Blender with the specified Python script
        command = [r"E:\Program Files\Blender Foundation\Blender 4.4\blender.exe", 
                   "--background", 
                   "--python", 
                   r"C:/Users/anchling/Documents/projects/neural-ultrasound-field\blenderInAndOut.py",
                   "--",
                   PATH]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output and error
        # print("Output:", result.stdout)
        # print("Error:", result.stderr)
        
        # num_scan = key.split("scan_")[1].split("_")[0]
        folder = os.path.join("c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records",args.record_folder,"sync","export")
        us_folder = os.path.join(folder,"us")
        nb_images = len(os.listdir(us_folder))


        slicesRendering(ckpt_path=ckpt_path,
                        output_path=folder,
                        number_of_slices=nb_images-1,
                        scale=1,
                        crop_info=None)
        
        file_list = ["us" + str(num) + ".jpg" for num in range(len(os.listdir(us_folder)))]
        bbox_path = os.path.join("c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records",args.record_folder,"bounding_box.json")
        if os.path.exists(bbox_path):
            with open(bbox_path, 'r') as f:
                bbox_info = json.load(f)
                roi2D = bbox_info["bounding_box"]
                strokes = bbox_info["strokes"]
            crop_tuple = (roi2D['x'],roi2D['y'],roi2D['x']+roi2D['width'],roi2D['y']+roi2D['height'])
            frames = [np.array(Image.open(os.path.join(us_folder, f)).crop(crop_tuple).convert('L')) for f in file_list]
        else:
            frames = [np.array(Image.open(os.path.join(us_folder, f)).convert('L')) for f in file_list]

        # add strokes to one frame with a selected size
        drawSize = 5
        for stroke in strokes:
            for point in stroke:
                x, y = point
                # draw a circle around the point
                cv2.circle(frames[0], (x, y), drawSize, (255, 0, 0), -1)


        # save all frames in a folder
        crop_folder = os.path.join(folder, "crop")
        os.makedirs(crop_folder, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(crop_folder, f"us{i}.jpg"))

                
        multi_frame_dicom_with_infosdat_positions(input_folder=us_folder,
                                                  dicom_path=os.path.join(out_folder,"original_crop.dcm"),
                                                  infos_file_path=os.path.join(folder,"infos.dat"),
                                                  prefix="us",
                                                  suffix=".jpg",
                                                  crop=roi2D,
                                                  )
        
        multi_frame_dicom_with_infosdat_positions(input_folder=us_folder,
                                                  dicom_path=os.path.join(out_folder,"original.dcm"),
                                                  infos_file_path=os.path.join(folder,"infos.dat"),
                                                  prefix="us",
                                                  suffix=".jpg",
                                                  )
        
        multi_frame_dicom_with_infosdat_positions(input_folder=os.path.join(out_folder,"reprod"),
                                                  dicom_path=os.path.join(out_folder,"reco.dcm"),
                                                  infos_file_path=os.path.join(folder,"infos.dat"),
                                                #   prefix="us",
                                                #   suffix=".jpg"
                                                  )
        
        print(80*"#")