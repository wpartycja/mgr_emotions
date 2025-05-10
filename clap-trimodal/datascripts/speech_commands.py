from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import defaultdict
import random


class SpeechCommandsText(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, split="train", max_len=128, samples_per_class=200):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples_per_class = samples_per_class

        self.original_dataset = load_dataset("speech_commands", "v0.01", split=split, trust_remote_code=True)

        # Define class groupings
        self.class_map = {
            "numbers": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "answer": ["yes", "no"],
            "directions": ["up", "down", "left", "right"],
            "animals": ["bird", "cat", "dog"]
        }
        
        self.label_to_group = self.__get_label_to_group()
        self.all_labels = sorted(self.class_map.keys())
        self.group2idx = {group: i for i, group in enumerate(self.all_labels)}
        self.filtered_dataset = self.__get_filtered_dataset(self.__get_grouped_data())

       
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

        return waveform, group, word

    def __get_label_to_group(self):
        label_to_group = {}
        for group, words in self.class_map.items():
            for w in words:
                label_to_group[w] = group
        return label_to_group
    
    def __get_grouped_data(self):
        grouped_data = defaultdict(list)
        for item in self.original_dataset:
            word = self.original_dataset.features['label'].int2str(item['label'])
            if word in self.label_to_group:
                grouped_data[self.label_to_group[word]].append(item)
        return grouped_data

    def __get_filtered_dataset(self, grouped_data):
        # Balance the dataset by sampling equal number per group
        filtered_dataset = []
        for group, items in grouped_data.items():
            if len(items) >= self.samples_per_class:
                sampled = random.sample(items, self.samples_per_class)
            else:
                sampled = items  # use all if not enough
            filtered_dataset.extend(sampled)
        return filtered_dataset

def speech_collate_fn(batch, tokenizer, cfg):
    audios, labels, text_inputs = zip(*batch)
    audios = torch.stack(audios)

    text_inputs = tokenizer(list(text_inputs), return_tensors="pt", padding=True, truncation=True)
    text_inputs = {key: val for key, val in text_inputs.items()}

   
    # Use the prompt template from Hydra config
    prompt_template = cfg.datasets.prompt_template
    label_texts = [prompt_template.format(label=label) for label in labels]

    class_text_inputs = tokenizer(
        label_texts, return_tensors="pt", padding=True, truncation=True, max_length=64
    )

    return audios, class_text_inputs, text_inputs
