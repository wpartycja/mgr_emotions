import os
import torch
import torchaudio
import torch.nn.functional as F

from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List, Tuple


class MultimodalSpeechDataset(Dataset, ABC):
    def __init__(self, 
                 data_dir: str, 
                 split: str,
                 cache_path: str,
                 sample_rate: int = 16000, 
                 max_length: int = 5):
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        cache_name = cache_path.split('.')[0]
        self.cache_path = f'{cache_name}_{split}.pkl'

        self.audio_paths = []
        self.transcripts = []
        self.labels = []

        self._load_metadata()

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        transcript = self.transcripts[idx]

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_length = self.sample_rate * self.max_length
        if waveform.size(1) < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.size(1)))
        else:
            waveform = waveform[:, :target_length]

        return waveform.squeeze(0), label, transcript

    @abstractmethod
    def _load_metadata(self) -> None:
        """Populates self.audio_paths, self.transcripts, self.labels"""
        pass
