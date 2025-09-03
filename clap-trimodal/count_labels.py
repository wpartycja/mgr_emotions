import torch
from collections import Counter

from datascripts.iemocap import IEMOCAPDataset
from datascripts.iemocap4cls import IEMOCAP4ClsDataset
from datascripts.meld import MELDDataset
from datascripts.meld4cls import MELD4ClsDataset


def count_labels(dataset_cls, data_dir, cache_path, splits, use_preprocessed=True, max_audio_length=8):
    """Count utterances per label across given splits for a dataset class."""
    label_counter = Counter()
    split_sizes = {}

    for split in splits:
        dataset = dataset_cls(
            data_dir=data_dir,
            split=split,
            cache_path=cache_path,
            use_preprocessed_audio=use_preprocessed,
            max_audio_length=max_audio_length,
        )
        split_sizes[split] = len(dataset)
        for _, label, _ in dataset:
            label_counter[label] += 1

    return split_sizes, label_counter


def main():
    use_preprocessed = True

    # IEMOCAP (10 classes)
    data_dir_iemocap = "data/processed_wav/iemocap" if use_preprocessed else "data/raw/iemocap"
    split_sizes_full, counts_full = count_labels(
        IEMOCAPDataset, data_dir_iemocap, "cache/iemocap_cache.pkl", ["train", "val", "test"], use_preprocessed
    )

    print("IEMOCAP (10 classes)")
    print("Split sizes:", split_sizes_full)
    print("\nUtterance count per class:")
    for label, count in sorted(counts_full.items()):
        print(f"{label}: {count}")
    print(f"Total: {sum(counts_full.values())}\n")

    # IEMOCAP (4 classes)
    split_sizes_4cls, counts_4cls = count_labels(
        IEMOCAP4ClsDataset, data_dir_iemocap, "cache/iemocap4cls_cache.pkl", ["train", "val", "test"], use_preprocessed
    )

    print("IEMOCAP (4 classes)")
    print("Split sizes:", split_sizes_4cls)
    print("\nUtterance count per class:")
    for label, count in sorted(counts_4cls.items()):
        print(f"{label}: {count}")
    print(f"Total: {sum(counts_4cls.values())}\n")

    # MELD (7 classes)
    data_dir_meld = "data/processed_pt/meld" if use_preprocessed else "data/processed_wav/meld"
    split_sizes_meld, counts_meld = count_labels(
        MELDDataset, data_dir_meld, "cache/meld_cache.pkl", ["train", "val", "test"], use_preprocessed
    )

    print("MELD (7 classes)")
    print("Split sizes:", split_sizes_meld)
    
    print("\nUtterance count per class:")
    for label, count in sorted(counts_meld.items()):
        print(f"{label}: {count}")
    print(f"Total: {sum(counts_meld.values())}\n")
    
    # MELD (4 classes)
    split_sizes_4cls, counts_4cls = count_labels(
        MELD4ClsDataset, data_dir_meld, "cache/meld4cls_cache.pkl", ["train", "val", "test"], use_preprocessed
    )

    print("MELD (4 classes)")
    print("Split sizes:", split_sizes_4cls)
    print("\nUtterance count per class:")
    for label, count in sorted(counts_4cls.items()):
        print(f"{label}: {count}")
    print(f"Total: {sum(counts_4cls.values())}\n")


if __name__ == "__main__":
    main()
