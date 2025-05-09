from datascripts.speech_commands import SpeechCommandsText, speech_collate_fn
from datascripts.ravdess import RAVDESSDatasetASR, ravdess_collate_fn


def load_dataset(cfg, tokenizer):
    name = cfg.datasets.name.lower()

    if name == "speech_commands":
        return SpeechCommandsText(
            tokenizer=tokenizer,
            split=cfg.datasets.split,
            samples_per_class=cfg.datasets.samples_per_class
        )
    elif name == "ravdess":
        return RAVDESSDatasetASR(
            data_dir=cfg.datasets.data_dir,
            cache_path=cfg.datasets.cache_file
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def load_dataset_and_collate_fn(cfg, tokenizer):
    dataset = load_dataset(cfg, tokenizer)
    name = cfg.datasets.name.lower()

    if name == "speech_commands":
        collate_fn = lambda batch: speech_collate_fn(batch, dataset, tokenizer)
    elif name == "ravdess":
        collate_fn = lambda batch: ravdess_collate_fn(batch, tokenizer, cfg)
    else:
        raise ValueError(f"No collate_fn for dataset: {name}")

    return dataset, collate_fn