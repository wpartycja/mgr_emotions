import glob
import os
from collections import defaultdict, Counter
from datascripts.base_dataset import MultimodalSpeechDataset
from typing import List, Tuple, Dict
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
from datascripts.prompt_utils import get_prompt
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple

from datascripts.base_dataset import MultimodalSpeechDataset


class IEMOCAPDataset(MultimodalSpeechDataset):
    """IEMOCAP audio-text dataset with deterministic train/val/test split."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        cache_path: str,
        sample_rate: int = 16_000,
        max_length: int = 5,
        use_preprocessed_audio: bool = True,
        train_rate: float = 0.80,
        eval_rate: float = 0.10,
    ):
        self.label_column = "Emotion"
        self.split = split.lower()
        self.use_preprocessed_audio = use_preprocessed_audio
        self.train_rate = train_rate
        self.eval_rate  = eval_rate
        self.all_labels = [
            "anger", 
            "happiness", 
            "excited", 
            "sadness", 
            "frustration", 
            "fear", 
            "surprise", 
            "other",
            "neutral",
            "disgust"
        ]
        
        self.all_labels_short = [
            "fru",
            "ang",
            "neu",
            "sad",
            "hap",
            "exc",
            "sur",
            "fea",
            "dis"
        ]
        
        self.labels_dict = {
            "fru": "frustration",
            "ang": "anger",
            "neu": "neutral",
            "sad": "sadness",
            "hap": "happiness",
            "exc": "excited",
            "sur": "surprise",
            "fea": "fear",
            "dis": "disgust"
        }


        super().__init__(data_dir, split, cache_path, sample_rate, max_length)


    def _load_metadata(self) -> None:
        """Populate self.audio_paths / self.labels / self.transcripts."""
        all_samples = self._gather_all_samples()      # List[(audio, lbl, txt)]
        split_samples = self._apply_split(all_samples)

        # finally populate the lists expected by MultimodalSpeechDataset
        self.audio_paths, self.labels, self.transcripts = map(list, zip(*split_samples))


    def _gather_all_samples(self) -> List[Tuple[str, str, str]]:
        """Walk the corpus once and return (audio_path, label, transcript) tuples."""
        sessions = sorted(d for d in os.listdir(self.data_dir) if d.startswith("Session"))
        samples: List[Tuple[str, str, str]] = []

        for session in sessions:
            s_path = Path(self.data_dir) / session
            label_dir = Path(self.data_dir) / session / "Labels"
            tran_dir = s_path / "transcriptions"
            wav_dir = s_path / "wav"

            utt_to_labels = self._collect_labels(label_dir)

            all_not_found_transcripts = 0

            for utt_id, majority_label in utt_to_labels.items():
                # Determine transcript filename
                if "script" in utt_id:
                    num_parts = 3
                elif "impro" in utt_id:
                    num_parts = 2
                else:
                    print(f"[Warning] Unknown utterance type for {utt_id}")
                    continue

                transcript_file = tran_dir / f"{'_'.join(utt_id.split('_')[:num_parts])}.txt"
                transcript = "[missing transcript]"
                found_transcript = False

                if transcript_file.exists():
                    with open(transcript_file) as tf:
                        for line in tf:
                            if utt_id in line and ":" in line:
                                parts = line.strip().split(":", 1)
                                if len(parts) == 2:
                                    transcript = parts[1].strip()
                                    found_transcript = True
                                    break

                if not found_transcript:
                    all_not_found_transcripts += 1
                    print(f"[Warning] Transcript not found for {utt_id} in {transcript_file}")

                # Determine audio path
                rel_folder = "_".join(utt_id.split('_')[:-1])
                wav_path = wav_dir / rel_folder / f"{utt_id}.wav"
                pt_path = wav_path.with_suffix(".pt")
                audio_path = pt_path if self.use_preprocessed_audio and pt_path.exists() else wav_path

                if not audio_path.exists():
                    print(f"[Missing audio] {audio_path}")
                    continue

                samples.append((str(audio_path), majority_label, transcript))

        print(f"Total missing transcripts in {self.split}: {all_not_found_transcripts}")
        samples.sort(key=lambda x: x[0])  # deterministic
        return samples

    def _collect_labels(self, label_dir: Path) -> dict:
        """Return utt_id → resolved_label based on primary line + fallback from annotator agreement."""
        valid_labels = set(l.lower() for l in self.all_labels_short)
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

                # Parse main line
                try:
                    parts = line.split("\t")
                    if len(parts) < 3:
                        i += 1
                        continue
                    utt_id = parts[1].strip()
                    primary_label = parts[2].strip().lower()
                except Exception:
                    i += 1
                    continue

                resolved_label = None
                fallback_lines = []
                i += 1

                # Collect following annotator lines until next block
                while i < len(lines) and not lines[i].startswith("["):
                    fallback_lines.append(lines[i].strip())
                    i += 1

                # Use primary label if valid
                if primary_label in valid_labels:
                    resolved_label = self.labels_dict[primary_label]
                else:
                    # Collect emotions from annotator lines (C-E2:, C-E3:, etc.)
                    votes = []
                    first_mentions = []

                    for fl in fallback_lines:
                        if fl.startswith("C-") and ":" in fl:
                            _, emostr = fl.split(":", 1)
                            emolist = [e.strip().strip(";").lower() for e in emostr.split(";") if e.strip()]
                            votes.extend(emolist)
                            if emolist:
                                first_mentions.append(emolist[0])

                    # Filter valid votes
                    votes = [v for v in votes if v in valid_labels]
                    first_mentions = [v for v in first_mentions if v in valid_labels]

                    if votes:
                        counts = Counter(votes).most_common()
                        max_count = counts[0][1]
                        tied = [l for l, c in counts if c == max_count]

                        if len(tied) == 1:
                            resolved_label = tied[0]
                        else:
                            # Use the one that was more often first
                            first_choice = Counter(first_mentions).most_common()
                            for l, _ in first_choice:
                                if l in tied:
                                    resolved_label = l
                                    break

                if resolved_label:
                    utt_to_label[utt_id] = resolved_label

        return utt_to_label


    def _collect_transcripts(self, tran_dir: Path) -> dict:
        """Return utt_id → transcript using proper filename logic."""
        transcripts = {}

        for file in tran_dir.glob("*.txt"):
            with open(file) as f:
                for line in f:
                    if ":" not in line:
                        continue
                    utt_id, txt = [p.strip() for p in line.split(":", 1)]
                    transcripts[utt_id] = txt

        return transcripts



    def _apply_split(self, data: List[Tuple[str, str, str]]
                     ) -> List[Tuple[str, str, str]]:
        """Deterministic train/val/test partition."""
        n_total  = len(data)
        n_train  = int(self.train_rate * n_total)
        n_val    = int(self.eval_rate  * n_total)
        train, val, test = data[:n_train // 4], data[n_train // 4 :n_train // 4 + n_val // 4], data[n_train+n_val:]

        if self.split == "train":
            return train
        elif self.split == "val":
            return val
        elif self.split == "test":
            return test
        else:
            raise ValueError(f"Invalid split name '{self.split}' (expect train|val|test)")


def iemocap_collate_fn(
        batch: List[Tuple[torch.Tensor, str, str]],
        tokenizer: PreTrainedTokenizer,
        cfg: DictConfig
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    
    max_text_length = getattr(cfg, "max_text_length", 64)

    waveforms, labels, transcripts = zip(*batch)
    waveforms = torch.stack(waveforms)
    
    class_texts = [get_prompt(label, cfg) for label in labels]

    # Tokenize transcripts
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