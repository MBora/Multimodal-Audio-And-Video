DURATION = 5.0
STRIDE = 0.2
SAVE_AS_MULTIPLE = 10 #saves in four digits btw
IMAGE_SIZE = (200,200)

import librosa
import cv2
import numpy as np
import os
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", "-i", help="Directory containing video files", type=str)
parser.add_argument("--out_dir", "-o", help="Directory to put resulting spectrograms in", type=str)
args = parser.parse_args()
inFolderName = args.in_dir
outFolderName = args.out_dir

if not os.path.isdir(outFolderName):
    os.mkdir(outFolderName)

for video_filename in os.listdir(inFolderName):
    file_name = '_'.join(video_filename.split('.')[:-1])
    audio_filename = f"{file_name}.wav"
    file_folder = os.path.join(outFolderName, file_name)
    audio_file_path = os.path.join(outFolderName, audio_filename)
    if not os.path.isdir(file_folder):
        os.mkdir(file_folder)

    vid = AudioSegment.from_file(os.path.join(inFolderName, video_filename), format='mp4')
    aud = vid.export(audio_file_path, format='wav')

    audio, sample_rate = librosa.load(audio_file_path)

    num_spectrograms = int(((len(audio) / sample_rate) - DURATION) / STRIDE)

    for i in range(num_spectrograms):
        spectrogram = audio[int(i * STRIDE * sample_rate) : int(((i  * STRIDE) + DURATION) * sample_rate)]

        spectrogram = librosa.core.stft(spectrogram)
        spectrogram = np.abs(spectrogram)
        spectrogram = librosa.core.amplitude_to_db(spectrogram, ref=np.max)

        spectrogram = (spectrogram - np.min(spectrogram)) / np.maximum(np.max(spectrogram) - np.min(spectrogram), np.finfo(float).eps)
        spectrogram = spectrogram * 255
        spectrogram = spectrogram.astype(np.uint8)

        spectrogram = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2RGB)

        spectrogram = cv2.resize(spectrogram, IMAGE_SIZE)

        save_format = '%04d_%04d'%(int(SAVE_AS_MULTIPLE * i * STRIDE),int(SAVE_AS_MULTIPLE * ((i  * STRIDE) + DURATION)))
        filename = f"{file_name}_{save_format}.jpg"
        filepath = os.path.join(file_folder, filename)
        cv2.imwrite(filepath, spectrogram)

    os.remove(audio_file_path)

print("Spectrograms Extracted!")
