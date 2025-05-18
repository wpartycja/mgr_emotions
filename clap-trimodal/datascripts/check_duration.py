import librosa
import numpy as np
from tqdm import tqdm
import glob
import os

audio_dirs = [
    '/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/meld_audio/test',
    '/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/meld_audio/train',
    '/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/meld_audio/val'
]

# Get all .wav files from all subdirectories
audio_paths = []
for dir_path in audio_dirs:
    audio_paths.extend(glob.glob(os.path.join(dir_path, '**', '*.wav'), recursive=True))

durations = []
for path in tqdm(audio_paths):
    try:
        duration = librosa.get_duration(filename=path)
        durations.append(duration)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Compute and print statistics
durations = np.array(durations)
print("Duration stats (in seconds):")
print(f"Min: {durations.min():.2f}")
print(f"Median (50th): {np.percentile(durations, 50):.2f}")
print(f"75th percentile: {np.percentile(durations, 75):.2f}")
print(f"90th percentile: {np.percentile(durations, 90):.2f}")
print(f"Max: {durations.max():.2f}")
print(f"Average: {durations.mean():.2f}")