import os
import torch
import pickle
import random

from collections import defaultdict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Union
from omegaconf import DictConfig
from torch import Tensor

from datasets import load_dataset


class SpeechCommandsText(Dataset):
    """Simplified version of speech commands dataset, with different classes and with two modalities: audio and text."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_len: int = 128,
        train_samples_per_class: int = 200,
        train_rate: float = 0.8,
        eval_rate: float = 0.1,
        seed: int = 42,
        cache_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.seed = seed
        self.train_rate = train_rate
        self.eval_rate = eval_rate
        self.cache_path = cache_path

        self.train_samples_per_class = train_samples_per_class
        self.total_samples_per_class = int(train_samples_per_class / train_rate)

        self.class_map = {
            "numbers": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "answer": ["yes", "no"],
            "directions": ["up", "down", "left", "right"],
            "animals": ["bird", "cat", "dog"],
        }

        self.label_to_group = self.__get_label_to_group()
        self.all_labels = sorted(self.class_map.keys())
        self.group2idx = {group: i for i, group in enumerate(self.all_labels)}
        self.group_labels = self.all_labels

        self.filtered_dataset = self._load_or_prepare_dataset()

    def __len__(self) -> int:
        return len(self.filtered_dataset)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor], int], Tuple[torch.Tensor, str, str]]:
        item = self.filtered_dataset[idx]
        waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)

        if waveform.size(0) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(0)))
        else:
            waveform = waveform[:16000]

        word = item["word"]
        group = item["group"]
        group_id = self.group2idx[group]

        if self.split == "train":
            text_inputs = self.tokenizer(
                word,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
            return waveform, text_inputs, group_id
        else:
            return waveform, group, word

    def __get_label_to_group(self) -> Dict[str, str]:
        return {word: group for group, words in self.class_map.items() for word in words}

    def __get_grouped_data(self, dataset: Dataset) -> Dict[str, List[Dict]]:
        grouped = defaultdict(list)
        for item in dataset:
            word = dataset.features["label"].int2str(item["label"])
            if word in self.label_to_group:
                group = self.label_to_group[word]
                grouped[group].append({**item, "word": word, "group": group})
        return grouped

    def _load_or_prepare_dataset(self) -> List[Dict]:
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading pre-transcribed cache from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                full_dataset = pickle.load(f)
        else:
            raw_dataset = load_dataset("speech_commands", "v0.01", split="train", trust_remote_code=True)
            grouped_data = self.__get_grouped_data(raw_dataset)

            full_dataset = []
            rng = random.Random(self.seed)

            for group, items in grouped_data.items():
                if len(items) < self.total_samples_per_class:
                    print(
                        f"Group '{group}' has only {len(items)} samples (needed {self.total_samples_per_class}). Using all."
                    )
                    sampled = items
                else:
                    sampled = rng.sample(items, self.total_samples_per_class)

                full_dataset.extend(sampled)

            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(full_dataset, f)

        return self.__split_cached_dataset(full_dataset)

    def __split_cached_dataset(self, full_dataset: List[Dict]) -> List[Dict]:
        grouped_data = defaultdict(list)
        for item in full_dataset:
            grouped_data[item["group"]].append(item)

        rng = random.Random(self.seed)
        split_data = []

        for group, items in grouped_data.items():
            rng.shuffle(items)

            n_total = len(items)
            n_train = int(self.train_rate * n_total)
            n_val = int(self.eval_rate * n_total)

            if self.split == "train":
                group_items = items[:n_train]
            elif self.split == "validation":
                group_items = items[n_train : n_train + n_val]
            elif self.split == "test":
                group_items = items[n_train + n_val :]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            split_data.extend(group_items)

        return split_data


def speech_collate_fn(
    batch: List[Tuple[Tensor, Dict[str, Tensor], int]],
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    dataset: SpeechCommandsText,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:

    audios, text_inputs, labels = zip(*batch)
    audios = torch.stack(audios)
    labels = torch.tensor(labels)

    text_inputs = {key: torch.stack([x[key] for x in text_inputs]) for key in text_inputs[0]}

    # Get label names from dataset (not cfg!)
    label_names = dataset.all_labels
    label_texts = [cfg.datasets.prompt_template.format(label=label_names[y]) for y in labels]

    class_text_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

    return audios, text_inputs, class_text_inputs
