# Zero-shot inference using tri-modal CLAP on RAVDESS (audio, text, or both vs. class descriptions)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from transformers import RobertaTokenizer
from datascripts.ravdess import RAVDESSDatasetWithASR
from model.clap_trimodal import CLAPTriModal
import torch.nn.functional as F
import random
from dotenv import load_dotenv
import os

load_dotenv()
access_token = os.getenv("HF_TOKEN")
model_path = "./weights/clap_trimodal_ravdess.pt"
cache_path = "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed/ravdess_cache.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label to prompt text
label2text = {
    "neutral": "This speech expresses neutrality.",
    "calm": "This speech sounds calm and peaceful.",
    "happy": "This speech sounds happy and joyful.",
    "sad": "This speech expresses sadness.",
    "angry": "This speech sounds angry.",
    "fearful": "This speech expresses fear.",
    "disgust": "This speech conveys disgust.",
    "surprised": "This speech sounds surprised."
}

all_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
emotion2idx = {emotion: idx for idx, emotion in enumerate(all_emotions)}
idx2emotion = {idx: emotion for emotion, idx in emotion2idx.items()}

labels_text = [label2text[emotion] for emotion in all_emotions]

# Load tokenizer and dataset
print("🔁 Loading tokenizer and dataset...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

test_dataset = RAVDESSDatasetWithASR(
    data_dir="/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/raw/ravdess",  
    preload_transcripts=True,
    save_cache=True,
    cache_path=cache_path
)


# Encode class descriptions
with torch.no_grad():
    model = CLAPTriModal("DistilHuBERT", "DistilRoBERTa", d_proj=128, access_token=access_token).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_tokens = tokenizer(labels_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
    class_embeds = model.class_text_encoder(class_tokens)  # (num_classes, d_proj)
    class_embeds = F.normalize(class_embeds, dim=-1)

# Evaluation
correct_audio, correct_text, correct_both = 0, 0, 0

total = len(test_dataset)
print(f"🔍 Running zero-shot inference on {total} samples...")
for i in range(total):
    waveform, label, transcript = test_dataset[i]
    waveform = waveform.unsqueeze(0).to(device)

    text_inputs = tokenizer(
        transcript,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    label_idx = torch.tensor([emotion2idx[label]]).to(device)

    with torch.no_grad():
        z_audio = model.audio_encoder(waveform)  # (1, d_proj)
        z_text = model.input_text_encoder(text_inputs)  # (1, d_proj)

        # Mode 1: audio only
        sims_audio = torch.matmul(z_audio, class_embeds.T)
        pred_audio = torch.argmax(sims_audio, dim=1)
        correct_audio += (pred_audio == label_idx).item()

        # Mode 2: text only
        sims_text = torch.matmul(z_text, class_embeds.T)
        pred_text = torch.argmax(sims_text, dim=1)
        correct_text += (pred_text == label_idx).item()

        # Mode 3: average of both
        z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
        sims_both = torch.matmul(z_avg, class_embeds.T)
        pred_both = torch.argmax(sims_both, dim=1)
        correct_both += (pred_both == label_idx).item()

print("\n🎯 Zero-Shot Inference Results:")
print(f"Total samples: {total}")
print(f"Accuracy (Audio only):  {100 * correct_audio / total:.2f}%")
print(f"Accuracy (Text only):   {100 * correct_text / total:.2f}%")
print(f"Accuracy (Audio + Text): {100 * correct_both / total:.2f}%")

# Random 5 samples for inspection
test_indices = random.sample(range(len(test_dataset)), 5)
test_samples = [test_dataset[i] for i in test_indices]

print("\n🔍 Inspecting 5 random samples...")
for i, (waveform, label, transcript) in enumerate(test_samples):
    waveform = waveform.unsqueeze(0).to(device)

    text_inputs = tokenizer(
        transcript,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    label_idx = emotion2idx[label]

    with torch.no_grad():
        z_audio = model.audio_encoder(waveform)  # (1, d_proj)
        z_text = model.input_text_encoder(text_inputs)  # (1, d_proj)

        print(f"\nSample {i+1} | Transcript: '{transcript}' | Ground truth: {label}")

        # Mode 1: Both
        z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
        sims_both = torch.matmul(z_avg, class_embeds.T)
        pred_both = torch.argmax(sims_both, dim=1).item()
        print(f"Prediction (audio + text): {idx2emotion[pred_both]}")

        # Mode 2: Audio only
        sims_audio = torch.matmul(z_audio, class_embeds.T)
        pred_audio = torch.argmax(sims_audio, dim=1).item()
        print(f"Prediction (audio only): {idx2emotion[pred_audio]}")

        # Mode 3: Text only
        sims_text = torch.matmul(z_text, class_embeds.T)
        pred_text = torch.argmax(sims_text, dim=1).item()
        print(f"Prediction (text only): {idx2emotion[pred_text]}")
