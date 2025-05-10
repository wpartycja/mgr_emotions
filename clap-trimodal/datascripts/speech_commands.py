from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import defaultdict
import random

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from collections import defaultdict
import random

class SpeechCommandsText(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split="train",
        max_len=128,
        train_samples_per_class=512,
        train_rate=0.8,
        eval_rate=0.1,
        seed=42,
    ):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split
        self.seed = seed
        self.train_rate = train_rate
        self.eval_rate = eval_rate

        # Calculate total samples per group needed
        self.desired_train_samples_per_class = train_samples_per_class
        self.total_samples_per_class = int(train_samples_per_class / train_rate)

        self.original_dataset = load_dataset("speech_commands", "v0.01", split="train", trust_remote_code=True)

        self.class_map = {
            "numbers": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "answer": ["yes", "no"],
            "directions": ["up", "down", "left", "right"],
            "animals": ["bird", "cat", "dog"],
        }

        self.label_to_group = self.__get_label_to_group()
        self.all_labels = sorted(self.class_map.keys())
        self.group2idx = {group: i for i, group in enumerate(self.all_labels)}

        self.filtered_dataset = self.__build_split_dataset()

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

        elif self.split == "validation":
            return waveform, group, word  # what your evaluate() expects

    def __get_label_to_group(self):
        return {word: group for group, words in self.class_map.items() for word in words}

    def __get_grouped_data(self):
        grouped = defaultdict(list)
        for item in self.original_dataset:
            word = self.original_dataset.features['label'].int2str(item['label'])
            if word in self.label_to_group:
                group = self.label_to_group[word]
                grouped[group].append(item)
        return grouped

    def __build_split_dataset(self):
        grouped_data = self.__get_grouped_data()
        split_data = []
        rng = random.Random(self.seed)

        for group, items in grouped_data.items():
            # Sample a fixed number of items per group
            if len(items) < self.total_samples_per_class:
                print(f"Group '{group}' has only {len(items)} samples (needed {self.total_samples_per_class}). Using all.")
                sampled = items
            else:
                sampled = rng.sample(items, self.total_samples_per_class)

            rng.shuffle(sampled)

            n_total = len(sampled)
            n_train = int(self.train_rate * n_total)
            n_val = int(self.eval_rate * n_total)

            if self.split == "train":
                group_items = sampled[:n_train]
            elif self.split == "validation":
                group_items = sampled[n_train:n_train + n_val]
            elif self.split == "test":
                group_items = sampled[n_train + n_val:]
            else:
                raise ValueError("Unknown split")

            split_data.extend(group_items)

        return split_data


def speech_collate_fn(batch, tokenizer, cfg, dataset):
    audios, text_inputs, labels = zip(*batch)
    audios = torch.stack(audios)
    labels = torch.tensor(labels)

    text_inputs = {key: torch.stack([x[key] for x in text_inputs]) for key in text_inputs[0]}

    # Get label names from dataset (not cfg!)
    label_names = dataset.all_labels
    label_texts = [cfg.datasets.prompt_template.format(label=label_names[y]) for y in labels]

    class_text_inputs = tokenizer(
        label_texts, return_tensors="pt", padding=True, truncation=True, max_length=64
    )

    return audios, text_inputs, class_text_inputs