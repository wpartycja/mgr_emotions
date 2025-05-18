import csv
import torch

from pathlib import Path
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig
from typing import List, Tuple, Dict
from datascripts.base_dataset import MultimodalSpeechDataset
from datascripts.prompt_utils import get_prompt


class MELDDataset(MultimodalSpeechDataset):
    def __init__(self, data_dir: str, split: str, cache_path: str, sample_rate: int = 16000, max_length: int = 5, use_preprocessed_audio: bool = True):
        self.label_column = "Emotion"

        self.all_labels = [
            "neutral",
            "joy",
            "surprise",
            "sadness",
            "anger",
            "fear",
            "disgust"
        ]
        
        self.use_preprocessed_audio = use_preprocessed_audio

        super().__init__(data_dir, split, cache_path, sample_rate, max_length)

    def _load_metadata(self):
        self.audio_paths = []
        self.transcripts = []
        self.labels = []

        csv_path = Path(self.data_dir) / f"{self.split}.csv"
        audio_dir = Path(self.data_dir) / self.split

        suffix = ".pt" if self.use_preprocessed_audio else ".wav"

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialog_id = row["Dialogue_ID"].strip()
                utt_id = row["Utterance_ID"].strip()
                base_name = f"dia{dialog_id}_utt{utt_id}{suffix}"
                audio_path = audio_dir / base_name

                if not audio_path.exists():
                    continue

                label = row[self.label_column].strip().lower()
                if label not in self.all_labels:
                    continue

                self.audio_paths.append(str(audio_path))
                self.transcripts.append(row["Utterance"].strip())
                self.labels.append(label)


def meld_collate_fn(
    batch: List[Tuple[torch.Tensor, str, str]],
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    max_text_length = getattr(cfg, "max_text_length", 64)

    waveforms, labels, transcripts = zip(*batch)
    waveforms = torch.stack(waveforms)

    # Prompt from label (e.g., "This person is feeling: {label}.")
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