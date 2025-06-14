from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from datascripts.speech_commands import SpeechCommandsText, speech_collate_fn
from datascripts.ravdess import RAVDESSDataset, ravdess_collate_fn
from datascripts.meld import MELDDataset, meld_collate_fn
from datascripts.iemocap import IEMOCAPDataset, iemocap_collate_fn



def get_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizer, split: str):
    name = cfg.dataset.name.lower()

    if name == "speech_commands":
        return SpeechCommandsText(
            tokenizer=tokenizer,
            split=split,
            train_samples_per_class=cfg.dataset.samples_per_class,
            cache_path=cfg.dataset.cache_file,
            max_audio_length=cfg.dataset.max_audio_length
        ) 
    elif name == "ravdess":
        return RAVDESSDataset(
            data_dir=cfg.dataset.data_dir,
            cache_path=cfg.dataset.cache_file,
            split=split,
            include_song=cfg.dataset.include_song,
            max_audio_length=cfg.dataset.max_audio_length
        )
    elif name == "meld":
        return MELDDataset(
            data_dir=cfg.dataset.data_dir,
            cache_path=cfg.dataset.cache_file,
            split=split,
            max_audio_length=cfg.dataset.max_audio_length
        )
    elif name == "iemocap":
        return IEMOCAPDataset(
            data_dir=cfg.dataset.data_dir,
            cache_path=cfg.dataset.cache_file,
            split=split,
            max_audio_length=cfg.dataset.max_audio_length
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_and_collate_fn(cfg: DictConfig, tokenizer: PreTrainedTokenizer, split: str = "train"):
    dataset = get_dataset(cfg, tokenizer, split)
    name = cfg.dataset.name.lower()

    if name == "speech_commands":
        collate_fn = lambda batch: speech_collate_fn(batch, tokenizer, cfg, dataset)
    elif name == "ravdess":
        collate_fn = lambda batch: ravdess_collate_fn(batch, tokenizer, cfg)
    elif name == "meld":
        collate_fn = lambda batch: meld_collate_fn(batch, tokenizer, cfg)
    elif name == "iemocap":
        collate_fn = lambda batch: iemocap_collate_fn(batch, tokenizer, cfg)
    else:
        raise ValueError(f"No collate_fn for dataset: {name}")

    return dataset, collate_fn
