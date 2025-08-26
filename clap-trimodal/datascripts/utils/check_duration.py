import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa

def print_duration_stats_from_dirs(dir_list, title: str):
    wav_paths = []
    for d in dir_list:
        wav_paths.extend(str(p) for p in Path(d).rglob("*.wav"))

    if not wav_paths:
        print(f"[warn] No .wav files found under: {dir_list}")
        return

    durations = []
    for p in tqdm(wav_paths, desc=f"Scanning {title}", unit="file"):
        try:
            durations.append(librosa.get_duration(filename=p))
        except Exception:
            try:
                y, sr = librosa.load(p, sr=None, mono=True)
                durations.append(len(y) / float(sr))
            except Exception as e2:
                print(f"[error] {p}: {e2}")

    durations = np.array(durations, dtype=float)
    if durations.size == 0:
        print(f"[warn] No durations computed for {title}.")
        return

    print("Duration stats (in seconds):")
    print(f"Min: {durations.min():.2f}")
    print(f"Median (50th): {np.percentile(durations, 50):.2f}")
    print(f"75th percentile: {np.percentile(durations, 75):.2f}")
    print(f"90th percentile: {np.percentile(durations, 90):.2f}")
    print(f"Max: {durations.max():.2f}")
    print(f"Average: {durations.mean():.2f}")


if __name__ == "__main__":
    # Example usage:

    # IEMOCAP
    IEMOCAP_WAV_DIRS = [
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/iemocap/Session1/wav",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/iemocap/Session2/wav",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/iemocap/Session3/wav",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/iemocap/Session4/wav",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/iemocap/Session5/wav",
    ]
    print_duration_stats_from_dirs(IEMOCAP_WAV_DIRS, title="IEMOCAP")

    print("\n")
    
    # MELD
    MELD_WAV_DIRS = [
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/meld/train",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/meld/val",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_wav/meld/test",
    ]
    print_duration_stats_from_dirs(MELD_WAV_DIRS, title="MELD")
