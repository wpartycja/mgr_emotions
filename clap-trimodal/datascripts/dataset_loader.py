from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from datascripts.meld import MELDDataset, meld_collate_fn
from datascripts.iemocap import IEMOCAPDataset, iemocap_collate_fn
from datascripts.iemocap4cls import IEMOCAP4ClsDataset
from datascripts.meld4cls import MELD4ClsDataset
from datascripts.soup_dataset import MELD_IEMOCAP_4ClsDataset



def get_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizer, split: str):
    name = cfg.dataset.name.lower()

    if name == "meld":
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
    elif name == 'iemocap4cls':
        return IEMOCAP4ClsDataset(
            data_dir=cfg.dataset.data_dir,
            cache_path=cfg.dataset.cache_file,
            split=split,
            max_audio_length=cfg.dataset.max_audio_length
        )
    elif name == 'meld4cls':
        return MELD4ClsDataset(
            data_dir=cfg.dataset.data_dir,
            cache_path=cfg.dataset.cache_file,
            split=split,
            max_audio_length=cfg.dataset.max_audio_length
        )
    elif name == "soup":
        return MELD_IEMOCAP_4ClsDataset(
            data_dirs=cfg.dataset.data_dirs,
            cache_path=cfg.dataset.cache_file,
            split=split,
            max_audio_length=cfg.dataset.max_audio_length,
            seed=cfg.dataset.get("seed", None),
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_and_collate_fn(cfg: DictConfig, tokenizer: PreTrainedTokenizer, split: str = "train"):
    dataset = get_dataset(cfg, tokenizer, split)
    name = cfg.dataset.name.lower()

    if name in ["meld", "meld4cls"]:
        collate_fn = lambda batch: meld_collate_fn(batch, tokenizer, cfg)
    elif name in ["iemocap", "iemocap4cls"]:
        collate_fn = lambda batch: iemocap_collate_fn(batch, tokenizer, cfg)
    elif name == "soup":
        collate_fn = lambda batch: meld_collate_fn(batch, tokenizer, cfg)
    else:
        raise ValueError(f"No collate_fn for dataset: {name}")

    return dataset, collate_fn
