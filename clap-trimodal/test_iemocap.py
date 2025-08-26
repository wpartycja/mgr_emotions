import torch
import random
import matplotlib.pyplot as plt
from collections import Counter

from datascripts.iemocap import IEMOCAPDataset

def main():
    use_preprocessed = True

    dataset = IEMOCAPDataset(
        data_dir="data/processed_wav/iemocap" if use_preprocessed else "data/raw/iemocap",
        split="train",
        cache_path="cache/iemocap_cache.pkl",
        use_preprocessed_audio=use_preprocessed,
        max_audio_length=8
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Count class occurrences
    label_counter = Counter()
    for _, label, _ in dataset:
        label_counter[label] += 1

    # Print utterance count per class
    print("\nUtterance count per class:")
    for label, count in sorted(label_counter.items()):
        print(f"{label}: {count}")



if __name__ == "__main__":
    main()
