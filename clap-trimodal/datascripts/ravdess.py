import os
import pickle
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

from base_dataset import MultimodalSpeechDataset


class RAVDESSDataset(MultimodalSpeechDataset):
    """RAVDESS dataset with audio + transcribed text using Whisper ASR."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        sample_rate: int = 16000,
        max_length: int = 5,
        train_rate: float = 0.8,
        eval_rate: float = 0.1,
        cache_path: str = None,
    ):
        self.train_rate = train_rate
        self.eval_rate = eval_rate
        self.cache_path = cache_path

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

        super().__init__(data_dir, split, sample_rate, max_length)

    def _load_metadata(self):
        self._collect_files()
        self._apply_split()
        self._transcribe()

    def _collect_files(self):
        """Collect all audio paths and emotion labels from filenames."""
        self.audio_paths = []
        self.labels = []

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if parts[0] == "03":  # audio-only modality
                        emotion_id = int(parts[2])
                        label = self.emotion_map.get(emotion_id)
                        if label:
                            self.audio_paths.append(os.path.join(root, file))
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
        elif self.split == "validation":
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
