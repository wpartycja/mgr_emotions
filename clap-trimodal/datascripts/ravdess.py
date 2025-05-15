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

from datascripts.prompt_utils import get_prompt

warnings.filterwarnings('ignore')

class RAVDESSDatasetASR(Dataset):
    """Ravdess Dataset with two modalities: audio and text."""

    def __init__(
        self,
        data_dir: str,
        cache_path: str,
        split: str,
        train_rate: float = 0.8,
        eval_rate: float = 0.1,
        sample_rate: int = 16000,
        max_length: int = 5,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.cache_path = cache_path  # to save or to load from
        self.split = split
        self.train_rate = train_rate
        self.eval_rate = eval_rate

        self.filepaths = []
        self.labels = []

        # from filenames to string
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

        self._collect_files()
        self._transcribe()

    def _collect_files(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if parts[0] == "03":
                        emotion_id = int(parts[2])
                        emotion_label = self.emotion_map.get(emotion_id, None)
                        if emotion_label is not None:
                            self.filepaths.append(os.path.join(root, file))
                            self.labels.append(emotion_label)

        # Apply split after collection
        data = list(zip(self.filepaths, self.labels))
        data.sort()  # Ensure deterministic order
        total = len(data)
        train_end = int(self.train_rate * total)
        val_end = int((self.train_rate + self.eval_rate) * total)

        if self.split == "train":
            data = data[:train_end]
        elif self.split == "validation":
            data = data[train_end:val_end]
        elif self.split == "test":
            data = data[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.filepaths, self.labels = zip(*data)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_length = self.sample_rate * self.max_length
        if waveform.size(1) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))
        else:
            waveform = waveform[:, :target_length]

        transcript = self.transcripts[idx]

        return waveform.squeeze(0), label, transcript

    def _transcribe_from_path(self, path: str) -> str:
        result = self.asr_pipeline(path, return_timestamps=False)
        return result["text"]

    def _transcribe(self) -> None:
        if os.path.exists(self.cache_path):
            print(f"Loading pre-transcribed cache from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                self.transcripts = pickle.load(f)
        else:
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

            print("Preloading transcripts with progress bar...")
            self.transcripts = []
            for path in tqdm(self.filepaths, desc="Transcribing audio"):
                transcript = self._transcribe_from_path(path)
                self.transcripts.append(transcript)

            if self.cache_path:
                print(f"Saving transcriptions to {self.cache_path}...")
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.transcripts, f)


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
