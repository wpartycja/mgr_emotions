# Tri-modal CLAP training script: audio ↔ input text ↔ class text

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from model.clap_trimodal import CLAPTriModal
from model.contrastive_loss import clip_contrastive_loss
from datascripts.ravdess import RAVDESSDatasetWithASR
from dotenv import load_dotenv
import os

load_dotenv()
access_token = os.getenv("HF_TOKEN")

# Configuration
config = {
    "audioenc_name": "DistilHuBERT",
    "textenc_name": "DistilRoBERTa",
    "d_proj": 128,
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 5,
    "beta": 1.0,   # Contrastive loss weight
    "min_temp": 0.01,
    "max_temp": 0.3,
}

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


# Initialize
print("Loading tokenizer and dataset...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
dataset = RAVDESSDatasetWithASR(
    "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/raw/ravdess",
    preload_transcripts=True,
    save_cache=True)

# num_classes = len(dataset.group2idx)

# DataLoader


def ravdess_collate_fn(batch, label2text_prompt, tokenizer, max_text_length=64):
    """
    Args:
        batch: list of (waveform, label_str, transcript)
        label2text_prompt: mapping like {"angry": "This speech expresses anger."}
        tokenizer: Huggingface tokenizer (for text and class text)
        max_text_length: maximum number of tokens
    Returns:
        waveforms: Tensor (batch_size, audio_length)
        input_text_inputs: Tokenized transcripts
        class_text_inputs: Tokenized class prompts
        labels: Tensor (batch_size)
    """
    waveforms, labels, transcripts = zip(*batch)

    waveforms = torch.stack(waveforms)

    # Build label2idx mapping
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    labels_idx = torch.tensor([label2idx[label] for label in labels])

    # Prepare class text descriptions
    class_texts = [label2text_prompt[label] for label in labels]

    # Tokenize transcripts and class descriptions
    input_text_inputs = tokenizer(
        list(transcripts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_length
    )

    class_text_inputs = tokenizer(
        list(class_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_length
    )

    return waveforms, input_text_inputs, class_text_inputs, labels_idx


# def collate_fn(batch):
#     audios, text_inputs, labels = zip(*batch)
#     audios = torch.stack(audios)
#     labels = torch.tensor(labels)
#     text_inputs = {key: torch.stack([x[key] for x in text_inputs]) for key in text_inputs[0]}

#     # Create class description text
#     label_texts = [f"This is the class: {train_dataset.group_labels[y]}" for y in labels]
#     class_text_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

#     return audios, text_inputs, class_text_inputs



train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: ravdess_collate_fn(batch, label2text, tokenizer)
)

# Model
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLAPTriModal(
    audioenc_name=config["audioenc_name"],
    textenc_name=config["textenc_name"],
    d_proj=config["d_proj"],
    access_token=access_token
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

# Training loop
print("Starting training...")
for epoch in range(config["epochs"]):
    model.train()
    total_loss, total_audio_text, total_audio_class, total_text_class = 0, 0, 0, 0

    for step, (audio, text_inputs, class_text_inputs, labels_idx) in enumerate(train_loader):
        audio = audio.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

        out = model(audio=audio, input_text=text_inputs, class_text=class_text_inputs)

        scale = out["contrastive_scale"]
        audio_embed = out["audio_embed"]
        text_embed = out["input_text_embed"]
        class_embed = out["class_text_embed"]

        # Contrastive losses between all 3 modalities (pairwise)
        loss_audio_text = clip_contrastive_loss(audio_embed, text_embed, scale, device=device)
        loss_audio_class = clip_contrastive_loss(audio_embed, class_embed, scale, device=device)
        loss_text_class = clip_contrastive_loss(text_embed, class_embed, scale, device=device)

        # Proportional loss weights
        losses = [loss_audio_text, loss_audio_class, loss_text_class]
        temperature = 0.5
        loss_tensor = torch.tensor([l.item() for l in losses])
        weights = torch.softmax(loss_tensor / temperature, dim=0)
        loss = sum(w * l for w, l in zip(weights, losses))
        

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_audio_text += loss_audio_text.item()
        total_audio_class += loss_audio_class.item()
        total_text_class += loss_text_class.item()

        if step % 10 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | A↔T: {loss_audio_text.item():.4f} | A↔C: {loss_audio_class.item():.4f} | T↔C: {loss_text_class.item():.4f}")
            print("Loss weights:", weights.tolist())
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"\n📘 Epoch {epoch+1}/{config['epochs']} | Avg Loss: {avg_loss:.4f}\n")

# Save model
torch.save(model.state_dict(), "clap_trimodal.pt")
print("Model saved to 'clap_trimodal.pt'")