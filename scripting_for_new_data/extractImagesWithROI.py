import cv2
import os
import shutil
import argparse
from extractImages import extract_frames, copy_images

def select_crop_rectangle(video_file, timecode=0.0):
    """Opens a frame at `timecode` and lets user select crop bounding box manually."""
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, timecode * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read frame from video")

    # Let user select ROI
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # ROI format: (x, y, w, h)
    x, y, w, h = roi
    return [y, x, y + h, x + w]  # Convert to [top, left, bottom, right] like your code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--applications_dir',
                        type=str,
                        help='root point of the applications cpp project (path/to/Applications)',
                        required=True
                        )
    parser.add_argument('--input_dir',
                        type=str,
                        help='Folder with videos of ultrasound Images',
                        required=True)
    
    args = parser.parse_args()

    output_folder = os.path.join(args.input_dir, "out")    # folder with extracted images
    
    # Extract frames from video
    video_names = [f for f in os.listdir(args.input_dir) if f.endswith(".wmv")]

    for video_name in video_names:
        input_video = os.path.join(args.input_dir, video_name) # In this case a .wmv located in the WMV folder within the input_dir
        frames_directory = extract_frames(video_file=input_video,
                                            output_dir=output_folder, 
                                            bounding_box=select_crop_rectangle(input_video), 
                                            timecodes=[], 
                                            crop=True, 
                                            export_fps=None)

        # Copy US to folder for Ultrasounds Importer
        US_dir = os.path.join(args.applications_dir, "data", "UltraSounds", "loop_US", "US")
        copy_images(frames_directory,US_dir)