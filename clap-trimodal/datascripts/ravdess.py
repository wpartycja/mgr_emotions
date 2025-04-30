import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
import pickle

class RAVDESSDatasetWithASR(Dataset):
    def __init__(self, data_dir, sample_rate=16000, max_length=5, preload_transcripts=True, save_cache=True, cache_path="/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/ravdess_cache.pkl"):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.preload_transcripts = preload_transcripts
        self.save_cache = save_cache
        self.cache_path = cache_path

        self.filepaths = []
        self.labels = []

        self.emotion_map = {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised"
        }

        self._collect_files()

        if preload_transcripts and os.path.exists(cache_path):
            print(f"⚡ Loading pre-transcribed cache from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.transcripts = pickle.load(f)
        else:
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

            if preload_transcripts:
                print("🔁 Preloading transcripts with progress bar...")
                self.transcripts = []
                for path in tqdm(self.filepaths, desc="Transcribing audio"):
                    transcript = self.transcribe(path)
                    self.transcripts.append(transcript)

                if save_cache:
                    print(f"💾 Saving transcriptions to {cache_path}...")
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.transcripts, f)

    def _collect_files(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split('-')
                    if parts[0] == "03" and parts[1] == "01":
                        emotion_id = int(parts[2])
                        emotion_label = self.emotion_map.get(emotion_id, None)
                        if emotion_label is not None:
                            self.filepaths.append(os.path.join(root, file))
                            self.labels.append(emotion_label)

    def __len__(self):
        return len(self.filepaths)

    def transcribe(self, path):
        result = self.asr_pipeline(path, return_timestamps=False)
        return result['text']

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_length = self.sample_rate * self.max_length
        if waveform.size(1) < target_length:
            padding = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_length]

        if self.preload_transcripts:
            transcript = self.transcripts[idx]
        else:
            transcript = self.transcribe(filepath)

        return waveform.squeeze(0), label, transcript



dataset = RAVDESSDatasetWithASR("/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/raw/ravdess", 
    preload_transcripts=True,
    save_cache=True,
    cache_path="/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/ravdess_cache.pkl"
)

waveform, label, transcript = dataset[0]
print(waveform.shape)
print(label)
print(transcript)