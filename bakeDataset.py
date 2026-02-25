from dataset import Dataset
import os
import argparse
import shutil

def get_arguments():
    """To obtain the different arguments from the parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        '-i', 
                        type=str, 
                        help='Folder with input files (images and infos.dat file with positions)',
                        default= "D:\\0-Code\\NeUF\\data\\simu_56\\us\\",
                        )
    parser.add_argument('--output', 
                        '-o', 
                        type=str, 
                        help='Path of output dataset pkl file',
                        default="D:\\0-Code\\NeUF\\data\\simu_56\\baked_simu_56.pkl",
                        )
    return parser.parse_args()

if __name__== "__main__":
    
    args = get_arguments()
    
    # datasetFolder = os.path.normpath(args.input_dir)
    # datasetFile = os.path.normpath(args.output)
    datasetFolder = "D:\\0-Code\\NeUF\\data\\cerebral_data\\Pre_traitement_echo_v2\\Recalage\\Patient0"
    datasetFile = "D:\\0-Code\\NeUF\\data\\cerebral_data\\Pre_traitement_echo_v2\\Recalage\\Patient0\\us_recal_small\\baked_patient0_recal_small.pkl"

    image_step = 1
    d = Dataset(datasetFolder,
                img_folder="us_recal_small",
                info_folder="us_recal_small",
                prefix="us",
                suffix=".jpg",
                reverse_quat=False,
                exclude_valid=False, 
                name= os.path.basename(datasetFolder),
                image_step=image_step)
    
    dataset_path = datasetFile.split('.')[0] + ".pkl"
    d.save(dataset_path)

    # copy dataset to input folder
    shutil.copy(dataset_path, os.path.join(datasetFolder, "dataset.pkl"))