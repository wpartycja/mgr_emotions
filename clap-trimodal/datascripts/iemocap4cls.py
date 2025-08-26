import os
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import torch

from datascripts.base_dataset import MultimodalSpeechDataset


class IEMOCAP4ClsDataset(MultimodalSpeechDataset):
    """
    IEMOCAP dataset filtered to 4 classes:
        - angry
        - happy (happiness + excited)
        - sad
        - neutral
    All other labels are discarded.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        cache_path: str,
        max_audio_length: int,
        use_preprocessed_audio: bool = True,
        train_rate: float = 0.80,
        eval_rate: float = 0.10,
    ):
        self.label_column = "Emotion"
        self.split = split.lower()
        self.use_preprocessed_audio = use_preprocessed_audio
        self.train_rate = train_rate
        self.eval_rate = eval_rate

        self.all_labels = ["angry", "happy", "sad", "neutral"]

        # map original IEMOCAP labels -> 4-class
        self.label_map = {
            "anger": "angry",
            "happiness": "happy",
            "excited": "happy",
            "sadness": "sad",
            "neutral": "neutral",
        }

        self.all_labels_short = [
            "ang", "hap", "exc", "sad", "neu"
        ]
        self.labels_dict = {
            "ang": "anger",
            "hap": "happiness",
            "exc": "excited",
            "sad": "sadness",
            "neu": "neutral",
        }

        super().__init__(data_dir, split, cache_path, max_audio_length)

    def _load_metadata(self) -> None:
        all_samples = self._gather_all_samples()
        split_samples = self._apply_split(all_samples)

        if not split_samples:
            raise RuntimeError("No samples found for split: " + self.split)

        self.audio_paths, self.labels, self.transcripts = map(list, zip(*split_samples))

    def _gather_all_samples(self) -> List[Tuple[str, str, str]]:
        sessions = sorted(d for d in os.listdir(self.data_dir) if d.startswith("Session"))
        samples: List[Tuple[str, str, str]] = []

        for session in sessions:
            s_path = Path(self.data_dir) / session
            label_dir = s_path / "Labels"
            tran_dir = s_path / "transcriptions"
            wav_dir = s_path / "wav"

            utt_to_labels = self._collect_labels(label_dir)

            for utt_id, orig_label in utt_to_labels.items():
                mapped_label = self.label_map.get(orig_label)
                if mapped_label not in self.all_labels:
                    continue  # skip unwanted labels

                transcript = self._get_transcript(utt_id, tran_dir)
                if transcript is None:
                    continue

                rel_folder = "_".join(utt_id.split('_')[:-1])
                wav_path = wav_dir / rel_folder / f"{utt_id}.wav"
                pt_path = wav_path.with_suffix(".pt")
                audio_path = pt_path if self.use_preprocessed_audio and pt_path.exists() else wav_path

                if not audio_path.exists():
                    print(f"[Missing audio] {audio_path}")
                    continue

                samples.append((str(audio_path), mapped_label, transcript))

        samples.sort(key=lambda x: x[0])
        return samples

    def _collect_labels(self, label_dir: Path) -> Dict[str, str]:
        valid_labels = set(self.all_labels_short)
        utt_to_label = {}

        for label_file in label_dir.glob("*.txt"):
            with open(label_file, encoding="utf-8") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or not line.startswith("["):
                    i += 1
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    i += 1
                    continue
                utt_id = parts[1].strip()
                primary_label = parts[2].strip().lower()

                resolved_label = None
                fallback_lines = []
                i += 1

                while i < len(lines) and not lines[i].startswith("["):
                    fallback_lines.append(lines[i].strip())
                    i += 1

                if primary_label in valid_labels:
                    resolved_label = self.labels_dict[primary_label]
                else:
                    votes = []
                    first_mentions = []
                    for fl in fallback_lines:
                        if fl.startswith("C-") and ":" in fl:
                            _, emostr = fl.split(":", 1)
                            emolist = [e.strip().strip(";").lower() for e in emostr.split(";") if e.strip()]
                            votes.extend([e for e in emolist if e in valid_labels])
                            if emolist:
                                first_mentions.append(emolist[0])

                    if votes:
                        counts = Counter(votes).most_common()
                        max_count = counts[0][1]
                        tied = [l for l, c in counts if c == max_count]

                        if len(tied) == 1:
                            resolved_label = self.labels_dict[tied[0]]
                        else:
                            first_choice = Counter(first_mentions).most_common()
                            for l, _ in first_choice:
                                if l in tied:
                                    resolved_label = self.labels_dict[l]
                                    break

                if resolved_label:
                    utt_to_label[utt_id] = resolved_label

        return utt_to_label

    def _get_transcript(self, utt_id: str, tran_dir: Path) -> str:
        if "script" in utt_id:
            num_parts = 3
        elif "impro" in utt_id:
            num_parts = 2
        else:
            return None

        transcript_file = tran_dir / f"{'_'.join(utt_id.split('_')[:num_parts])}.txt"
        if not transcript_file.exists():
            return None

        with open(transcript_file) as tf:
            for line in tf:
                if utt_id in line and ":" in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
        return None

    def _apply_split(self, data: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        n_total = len(data)
        n_train = int(self.train_rate * n_total)
        n_val = int(self.eval_rate * n_total)

        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]

        if self.split == "train":
            return train
        elif self.split == "val":
            return val
        elif self.split == "test":
            return test
        else:
            raise ValueError(f"Invalid split name '{self.split}' (expect train|val|test)")
