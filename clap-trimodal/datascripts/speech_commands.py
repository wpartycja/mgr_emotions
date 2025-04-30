from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import defaultdict
import random

class SpeechCommandsText(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, split="train", sample_rate=16000, max_len=128, samples_per_class=200):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_len = max_len

        self.original_dataset = load_dataset("speech_commands", "v0.01", split=split, trust_remote_code=True)

        # Define class groupings
        self.class_map = {
            "numbers": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "answer": ["yes", "no"],
            "directions": ["up", "down", "left", "right"],
            "animals": ["bird", "cat", "dog"]
        }

        self.label_to_group = {}
        for group, words in self.class_map.items():
            for w in words:
                self.label_to_group[w] = group

        self.group_labels = sorted(self.class_map.keys())
        self.group2idx = {group: i for i, group in enumerate(self.group_labels)}

        grouped_data = defaultdict(list)
        for item in self.original_dataset:
            word = self.original_dataset.features['label'].int2str(item['label'])
            if word in self.label_to_group:
                grouped_data[self.label_to_group[word]].append(item)

        # Balance the dataset by sampling equal number per group
        self.filtered_dataset = []
        for group, items in grouped_data.items():
            if len(items) >= samples_per_class:
                sampled = random.sample(items, samples_per_class)
            else:
                sampled = items  # use all if not enough
            self.filtered_dataset.extend(sampled)

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        item = self.filtered_dataset[idx]
        waveform = torch.tensor(item['audio']['array'], dtype=torch.float32)

        if waveform.size(0) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(0)))
        else:
            waveform = waveform[:16000]

        word = self.original_dataset.features['label'].int2str(item['label'])
        group = self.label_to_group[word]
        group_id = self.group2idx[group]

        text_inputs = self.tokenizer(
            word,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return waveform, text_inputs, group_id