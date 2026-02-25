import cv2
import os
import shutil
import argparse

def extract_frames(video_file, output_dir, bounding_box, timecodes, crop=False , export_fps=None):
    """Extract frames from a video with a certain bounding box in between two timecodes at a certain frame_rate"""
    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(video_fps)

    if len(timecodes) == 2:
        start_time, end_time = timecodes[0],timecodes[1]
    else:
        start_time, end_time = 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps

    start_ms = start_time * 1000
    end_ms = end_time * 1000

    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_directory = output_dir + f"/{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
    current_ms = start_ms
    frame_id = 0

    step = 1000 / export_fps if export_fps is not None else 1000 / video_fps

    while current_ms <= end_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
        ret, frame = cap.read()

        if not ret:
            break
        
        if crop:
            frame = frame[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

        output_file = f"{output_directory}/us{frame_id}.jpg"
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to Grayscale
        cv2.imwrite(output_file, grayscale_frame)
        frame_id += 1

        current_ms += step

    cap.release()
    cv2.destroyAllWindows()

    return output_directory

def copy_images(input_dir, output_dir):
    """Copy an entire folder of images to another location"""
    os.makedirs(output_dir, exist_ok=True)

    for entry in os.scandir(input_dir):
        if entry.is_file():
            shutil.copy(entry.path, os.path.join(output_dir, entry.name))

def compute_metrics(original_bb, crop_bb):
    """Compute the different scale factors for the cropped volume to be put back to the original volume"""
    
    height_orig, width_orig = original_bb[2]-original_bb[0], original_bb[3]-original_bb[1] 
    print(f"Original --> Width : {width_orig} | Height : {height_orig}")

    height_crop, width_crop = crop_bb[2]-crop_bb[0], crop_bb[3]-crop_bb[1]
    print(f"Crop     --> Width : {width_crop} | Height : {height_crop}")

    delta_h = crop_bb[0] - original_bb[0]
    delta_w = crop_bb[1] - original_bb[1]
    print(f"delta --> W : {delta_w} | H : {delta_h}")

    scale_h, scale_w = height_crop / height_orig, width_crop / width_orig
    print(f"scale --> W : {scale_w} | H : {scale_h}")

    print(80*"#")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--applications_dir',
                        type=str,
                        help='root point of the applications cpp project (path/to/Applications)',
                        required=True)
    parser.add_argument('--input_dir',
                        type=str,
                        default='./US_data',
                        help='Folder with videos of ultrasound Images',
                        required=False)
    args = parser.parse_args()

    output_folder = os.path.join(args.input_dir, "test")    # folder with extracted images
    
    cropping = True # Selection if output is cropped or not
    if cropping:
        output_folder = os.path.join(output_folder,"cropped")
    else:
        output_folder = os.path.join(output_folder,"fullscreen")
    
    # Cropping infos for entire US images
    imgs_info = {
        1 : [76,47,496,755],         # 2.5cm
        2 : [76,79,496,721],    # 3cm
        3 : [76,188,496,612],   # 4cm
    }

    # Cropping infos to focus on specific RoI
    crop_info = {
        "01" : {"bb":[80,249,258,559],
                "cat":1,
                "timecodes": [2,14]},         # 2.5cm
        "03" : {"bb":[88,300,301,635],
                "cat":1,
                "timecodes": [0,10]},         # 2.5cm
        "07" : {"bb":[107,423,299,621],
                "cat":2,
                "timecodes": []},    # 3cm
        "11" : {"bb":[259,330,430,512],
                "cat":3,
                "timecodes": []},   # 4cm
    }

    for num in ["01"]:#,"03","07","11"]:
        print(f"For scan numero : {num}")
        # Extract frames from video
        input_video = os.path.join(args.input_dir, "WMV", "Image"+num+".wmv") # In this case a .wmv located in the WMV folder within the input_dir
        frames_directory = extract_frames(video_file=input_video,
                                          output_dir=output_folder, 
                                          bounding_box=crop_info[num]["bb"], 
                                          timecodes=crop_info[num]["timecodes"], 
                                          crop=cropping, 
                                          export_fps=None)

        # Copy US to folder for Ultrasounds Importer
        US_dir = os.path.join(args.applications_dir, "data", "UltraSounds", "scan_" + num)
        copy_images(frames_directory,US_dir)

        # Compute scalefacor, width and height deltas...
        category = crop_info[num]["cat"]
        compute_metrics(original_bb=imgs_info[category],
                     crop_bb=crop_info[num]["bb"])