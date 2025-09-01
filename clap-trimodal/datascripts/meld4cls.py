# datascripts/meld4cls.py
import csv
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig

from datascripts.base_dataset import MultimodalSpeechDataset
from datascripts.prompt_utils import get_prompt


class MELD4ClsDataset(MultimodalSpeechDataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 cache_path: str,
                 max_audio_length: int,
                 use_preprocessed_audio: bool = True):
        self.label_column = "Emotion"
        self.split = split.lower()
        self.use_preprocessed_audio = use_preprocessed_audio

        self.all_labels = ["angry", "happy", "sad", "neutral"]
        # raw -> 4-class mapping
        self.label_map: Dict[str, Optional[str]] = {
            "anger":   "angry",
            "joy":     "happy",
            "sadness": "sad",
            "neutral": "neutral",
            # drop these by mapping to None (or just leave unmapped)
            "disgust": None,
            "fear":    None,
            "surprise": None,
        }

        super().__init__(data_dir, split, cache_path, max_audio_length)

    def _load_metadata(self):
        self.audio_paths: List[str] = []
        self.transcripts: List[str] = []
        self.labels: List[str] = []

        csv_path = Path(self.data_dir) / f"{self.split}.csv"   # expects train.csv / val.csv / test.csv
        audio_dir = Path(self.data_dir) / self.split           # and train/val/test folders with audio

        suffix = ".pt" if self.use_preprocessed_audio else ".wav"

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialog_id = row["Dialogue_ID"].strip()
                utt_id = row["Utterance_ID"].strip()
                base_name = f"dia{dialog_id}_utt{utt_id}{suffix}"
                audio_path = audio_dir / base_name
                if not audio_path.exists():
                    continue

                raw_label = row[self.label_column].strip().lower()
                mapped = self.label_map.get(raw_label)
                if mapped not in self.all_labels:
                    continue  # discard non-4-class labels

                self.audio_paths.append(str(audio_path))
                self.transcripts.append(row["Utterance"].strip())
                self.labels.append(mapped)


def meld4cls_collate_fn(
    batch: List[Tuple[torch.Tensor, str, str]],
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    waveforms, labels, transcripts = zip(*batch)
    waveforms = torch.stack(waveforms)

    class_texts = [get_prompt(label, cfg) for label in labels]

    input_text_inputs = tokenizer(
        list(transcripts), return_tensors="pt", padding=True, truncation=True,
        max_length=cfg.dataset.max_text_length,
    )
    class_text_inputs = tokenizer(
        class_texts, return_tensors="pt", padding=True, truncation=True,
        max_length=cfg.dataset.max_text_length,
    )
    return waveforms, input_text_inputs, class_text_inputs
