import glob
import os
from collections import defaultdict, Counter
from datascripts.base_dataset import MultimodalSpeechDataset
from typing import List, Tuple, Dict
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
from datascripts.prompt_utils import get_prompt



class IEMOCAPDataset(MultimodalSpeechDataset):
    def __init__(self, data_dir: str, split: str, cache_path: str, sample_rate: int = 16000, max_length: int = 5, use_preprocessed_audio: bool = True):
        self.label_column = "Emotion"
        self.all_labels = [
            "anger", 
            "happiness", 
            "excitement", 
            "sadness", 
            "frustration", 
            "fear", 
            "surprise", 
            "other",
            "neutral state"
        ]
        self.use_preprocessed_audio = use_preprocessed_audio
        super().__init__(data_dir, split, cache_path, sample_rate, max_length)

    import os, glob
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
            "excitement", 
            "sadness", 
            "frustration", 
            "fear", 
            "surprise", 
            "other",
            "neutral state"
        ]

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
            cat_dir = s_path / "Categorical"
            tran_dir = s_path / "transcriptions"
            wav_dir = s_path / "wav"

            utt_to_labels = self._collect_labels(cat_dir)

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

    def _collect_labels(self, cat_dir: Path) -> dict:
        """Return utt_id → majority_label dict for a session."""
        tmp: defaultdict[str, List[str]] = defaultdict(list)

        for cat_file in cat_dir.glob("*_e*_cat.txt"):
            with open(cat_file) as f:
                for line in f:
                    if ":" not in line:
                        continue
                    utt_id, label_part = [p.strip() for p in line.split(":", 1)]
                    raw = [
                        l.replace(";", "").replace("()", "").strip().lower()
                        for l in label_part.split(":") if l.strip() and l not in {"()", ";"}
                    ]
                    tmp[utt_id].extend(raw)

        majority: dict[str, str] = {}
        for utt_id, lab_list in tmp.items():
            lab_list = [l for l in lab_list if l not in {"xxx", "other", "not applicable"}]
            if not lab_list:
                continue
            counts = Counter(lab_list).most_common()
            max_cnt = counts[0][1]
            tied    = sorted(l for l, c in counts if c == max_cnt)
            majority[utt_id] = tied[0]  # alphabetical tie-break
        return majority


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
        train, val, test = data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]

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