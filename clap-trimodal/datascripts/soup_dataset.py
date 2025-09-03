# datascripts/meld_iemocap4cls.py
import random
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import torch

from datascripts.meld4cls import MELD4ClsDataset
from datascripts.iemocap4cls import IEMOCAP4ClsDataset


from omegaconf import DictConfig, OmegaConf


class MELD_IEMOCAP_4ClsDataset(Dataset):
    """
    Combined dataset for MELD (4-class) + IEMOCAP (4-class).
    Concatenates and shuffles samples from both datasets.
    """

    def __init__(
        self,
        data_dirs: Dict[str, str],
        split: str,
        cache_path: str,
        max_audio_length: int,
        use_preprocessed_audio: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split.lower()
        self.use_preprocessed_audio = use_preprocessed_audio
        self.max_audio_length = max_audio_length


        self.all_labels = ["angry", "happy", "sad", "neutral"]
        
        if isinstance(data_dirs, DictConfig):
            data_dirs = OmegaConf.to_container(data_dirs, resolve=True)

        required_keys = {"meld", "iemocap"}
        if not isinstance(data_dirs, dict):
            raise TypeError(f"data_dirs must be a dict with keys {required_keys}, got {type(data_dirs)}")
        if set(data_dirs.keys()) != required_keys:
            raise ValueError(f"data_dirs must contain exactly these keys: {required_keys}, got {set(data_dirs.keys())}")

        meld_ds = MELD4ClsDataset(
            data_dir=data_dirs["meld"],
            split=self.split,
            cache_path=cache_path,
            max_audio_length=max_audio_length,
            use_preprocessed_audio=use_preprocessed_audio,
        )
        iemocap_ds = IEMOCAP4ClsDataset(
            data_dir=data_dirs["iemocap"],
            split=self.split,
            cache_path=cache_path,
            max_audio_length=max_audio_length,
            use_preprocessed_audio=use_preprocessed_audio,
        )

        self.audio_paths = meld_ds.audio_paths + iemocap_ds.audio_paths
        self.labels = meld_ds.labels + iemocap_ds.labels
        self.transcripts = meld_ds.transcripts + iemocap_ds.transcripts
        self.dataset_ids = ["meld"] * len(meld_ds) + ["iemocap"] * len(iemocap_ds)

        random.seed(seed)
        indices = list(range(len(self.audio_paths)))
        random.shuffle(indices)

        self.audio_paths = [self.audio_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.transcripts = [self.transcripts[i] for i in indices]
        self.dataset_ids = [self.dataset_ids[i] for i in indices]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        transcript = self.transcripts[idx]
        return self._load_audio(audio_path), label, transcript

    def _load_audio(self, path: str) -> torch.Tensor:
        import torchaudio
        if path.endswith(".pt"):
            wav = torch.load(path)
        else:
            wav, sr = torchaudio.load(path)

        if wav.ndim == 2 and wav.size(0) == 1:
            wav = wav.squeeze(0) 
        elif wav.ndim == 3 and wav.size(0) == 1 and wav.size(1) == 1:
            wav = wav.squeeze(0).squeeze(0)  

        return wav
