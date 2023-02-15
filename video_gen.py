STRIDE = 0.2  #Reads 5 frames per second
SAVE_AS_MULTIPLE = 10 #saves in four digits btw

import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', '-i', help="Directory containing video files", type=str)
parser.add_argument('--out_dir', '-o', help="Directory to put resulting frames in", type=str)
args = parser.parse_args()
inFolderName = args.in_dir
outFolderName = args.out_dir

if not os.path.isdir(outFolderName):
    os.mkdir(outFolderName)

for video_filename in os.listdir(inFolderName):
    video_path = os.path.join(inFolderName, video_filename)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_skip = int(fps * STRIDE)
    frame_number = 0
    success, frame = video_capture.read()
    video_name = '_'.join(video_filename.split('.')[:-1])
    file_folder = os.path.join(outFolderName, video_name)
    if not os.path.isdir(file_folder):
        os.mkdir(file_folder)
    while success:
        if frame_number % frame_skip == 0:
            frame_name = f"{video_name}_{'%04d' % int(frame_number * SAVE_AS_MULTIPLE / fps)}.jpg"
            frame_path = os.path.join(file_folder, frame_name)
            cv2.imwrite(frame_path, frame)
        success, frame = video_capture.read()
        frame_number += 1

print("Frames Extracted!")
