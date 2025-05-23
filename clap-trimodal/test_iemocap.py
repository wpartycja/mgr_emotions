import torch
import random
import matplotlib.pyplot as plt

from datascripts.iemocap import IEMOCAPDataset

def main():
    # Toggle this to True/False to test both modes
    use_preprocessed = True

    dataset = IEMOCAPDataset(
        data_dir="data/processed/iemocap" if use_preprocessed else "data/raw/iemocap",
        split="train",
        cache_path="cache/iemocap_cache.pkl",
        use_preprocessed_audio=use_preprocessed,
        sample_rate=16000,
        max_length=5
    )

    print(f"Dataset size: {len(dataset)} samples")
     
    # Pick a few random samples
    for i in random.sample(range(len(dataset)), 3):
        waveform, label, transcript = dataset[i]
        print(f"\nSample {i}")
        print(f"Label: {label}")
        print(f"Transcript: {transcript}")
        print(f"Waveform shape: {waveform.shape}")
        
if __name__ == "__main__":
    main()
