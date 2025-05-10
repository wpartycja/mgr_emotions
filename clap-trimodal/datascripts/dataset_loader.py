from datascripts.speech_commands import SpeechCommandsText, speech_collate_fn
from datascripts.ravdess import RAVDESSDatasetASR, ravdess_collate_fn


def get_dataset(cfg, tokenizer, split):
    name = cfg.datasets.name.lower()

    if name == "speech_commands":
        return SpeechCommandsText(
            tokenizer=tokenizer,
            split=split,
            train_samples_per_class=cfg.datasets.samples_per_class,
            cache_path=cfg.datasets.cache_file,
        )
    elif name == "ravdess":
        return RAVDESSDatasetASR(
            data_dir=cfg.datasets.data_dir,
            cache_path=cfg.datasets.cache_file,
            split=split,
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_and_collate_fn(cfg, tokenizer, split="train"):
    dataset = get_dataset(cfg, tokenizer, split)
    name = cfg.datasets.name.lower()

    if name == "speech_commands":
        collate_fn = lambda batch: speech_collate_fn(batch, tokenizer, cfg, dataset)
    elif name == "ravdess":
        collate_fn = lambda batch: ravdess_collate_fn(batch, tokenizer, cfg)
    else:
        raise ValueError(f"No collate_fn for dataset: {name}")

    return dataset, collate_fn
