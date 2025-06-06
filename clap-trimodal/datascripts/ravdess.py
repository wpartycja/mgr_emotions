import os
import torch
import torchaudio
import pickle
import warnings

from transformers import pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig
from pathlib import Path

from datascripts.prompt_utils import get_prompt
from datascripts.base_dataset import MultimodalSpeechDataset


warnings.filterwarnings('ignore')


class RAVDESSDataset(MultimodalSpeechDataset):
    """RAVDESS dataset with audio + transcribed text using Whisper ASR."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        cache_path,
        max_audio_length: int,
        sample_rate: int = 16000,
        train_rate: float = 0.8,
        eval_rate: float = 0.1,
        include_song: bool = True
    ):
        self.train_rate = train_rate
        self.eval_rate = eval_rate
        
        self.include_song = include_song

        self.emotion_map = {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised",
        }
        
        self.all_labels = sorted(self.emotion_map.values())

        super().__init__(data_dir, split, cache_path, max_audio_length, sample_rate)

    def _load_metadata(self):
        self._collect_files()
        self._apply_split()
        self._transcribe()

    def _collect_files(self):
        """Recursively collect all .wav files and parse emotion from filename."""
        self.audio_paths = []
        self.labels = []

        data_dir = Path(self.data_dir)

        subdirs = []
        if self.include_song:
            subdirs.append(data_dir / "Audio_Song_Actors_01-24")
        subdirs.append(data_dir / "Audio_Speech_Actors_01-24")

        for base_dir in subdirs:
            for file in base_dir.rglob("*.wav"):
                parts = file.stem.split("-")
                if parts[0] != "03":
                    continue  # skip unexpected files
                emotion_id = int(parts[2])
                label = self.emotion_map.get(emotion_id)
                if label:
                    self.audio_paths.append(str(file))
                    self.labels.append(label)

    def _apply_split(self):
        """Split data deterministically into train/val/test."""
        data = list(zip(self.audio_paths, self.labels))
        data.sort()

        total = len(data)
        train_end = int(self.train_rate * total)
        val_end = int((self.train_rate + self.eval_rate) * total)

        if self.split == "train":
            split_data = data[:train_end]
        elif self.split == "val":
            split_data = data[train_end:val_end]
        elif self.split == "test":
            split_data = data[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.audio_paths, self.labels = zip(*split_data)

    def _transcribe(self):
        """Transcribe audio using Whisper ASR, or load from cache."""
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading cached transcripts from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                self.transcripts = pickle.load(f)
        else:
            print("Running Whisper ASR on RAVDESS audio...")
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
            self.transcripts = []

            for path in tqdm(self.audio_paths, desc="Transcribing"):
                result = self._transcribe_from_path(path)
                self.transcripts.append(result)

            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.transcripts, f)

    def _transcribe_from_path(self, path: str) -> str:
        """Run Whisper on a single audio path."""
        result = self.asr_pipeline(path, return_timestamps=False)
        return result["text"]


def ravdess_collate_fn(
    batch: List[Tuple[torch.Tensor, str, str]],
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    max_text_length = getattr(cfg, "max_text_length", 64)

    waveforms, labels, transcripts = zip(*batch)
    waveforms = torch.stack(waveforms)

    class_texts = [get_prompt(label, cfg) for label in labels]

    input_text_inputs = tokenizer(
        list(transcripts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_length,
    )

    class_text_inputs = tokenizer(
        class_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_length,
    )

    return waveforms, input_text_inputs, class_text_inputs