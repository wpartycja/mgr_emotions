import os
import torch
import torchaudio
import torch.nn.functional as F

from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List, Tuple

SAMPLE_RATE = 16_000  # matching for Wav2Vec2 / HuBERT / DistilHuBERT

class MultimodalSpeechDataset(Dataset, ABC):
    def __init__(self, 
                 data_dir: str, 
                 split: str,
                 cache_path: str,
                 max_audio_length: int,
                 sample_rate: int = SAMPLE_RATE):
        
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        cache_name = cache_path.split('.')[0]
        self.cache_path = f'{cache_name}_{split}.pkl'

        self.audio_paths = []
        self.transcripts = []
        self.labels = []

        self._load_metadata()

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        transcript = self.transcripts[idx]

        if path.endswith(".pt"):
            waveform = torch.load(path)
        else:
            waveform, sr = torchaudio.load(path)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Clip or pad to target length
            target_length = self.sample_rate * self.max_audio_length
            if waveform.size(1) < target_length:
                pad_len = target_length - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_len))
            else:
                waveform = waveform[:, :target_length]

        return waveform.squeeze(0), label, transcript

    @abstractmethod
    def _load_metadata(self) -> None:
        """Populates self.audio_paths, self.transcripts, self.labels"""
        pass
