from datascripts.speech_commands import SpeechCommandsText, speech_collate_fn
from datascripts.ravdess import RAVDESSDatasetASR, ravdess_collate_fn

def get_dataset_and_collate(cfg, tokenizer):
    name = cfg.datasets.name.lower()

    if name == "speech_commands":
        dataset = SpeechCommandsText(tokenizer, split=cfg.datasets.split, samples_per_class=cfg.datasets.samples_per_class)
        collate_fn = lambda batch: speech_collate_fn(batch, dataset, tokenizer)
    elif name == "ravdess":
        dataset = RAVDESSDatasetASR(
            data_dir=cfg.paths.data_dir,
            preload_transcripts=True,
            cache_path=cfg.paths.cache_file
        )
        collate_fn = lambda batch: ravdess_collate_fn(batch, tokenizer, cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataset, collate_fn
