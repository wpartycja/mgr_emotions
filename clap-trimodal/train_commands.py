# Tri-modal CLAP training script: audio ↔ input text ↔ class text

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AutoModel
from model.clap_trimodal import CLAPTriModal
from model.contrastive_loss import clip_contrastive_loss
from datascripts.speech_commands import SpeechCommandsText
from dotenv import load_dotenv
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datascripts.loader import get_dataset_and_collate


load_dotenv()
access_token = os.getenv("HF_TOKEN")



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print full config for logging
    
    
    # Initialize
    print("Loading tokenizer and dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset, collate_fn = get_dataset_and_collate(cfg, tokenizer)
    print("Dataset:", dataset)
    print("Collate fn:", collate_fn)
    print("Length:", len(dataset))
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


    # Model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLAPTriModal(
        audioenc_name=cfg.model.audio_encoder,
        textenc_name=cfg.model.text_encoder,
        d_proj=cfg.model.d_proj,
        access_token=access_token
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.max_temp)

    # Training loop
    print("Starting training...")
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss, total_audio_text, total_audio_class, total_text_class = 0, 0, 0, 0

        for step, (audio, text_inputs, class_text_inputs) in enumerate(dataloader):
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
        avg_loss = total_loss / len(dataloader)
        print(f"\n📘 Epoch {epoch+1}/{cfg.train.epochs} | Avg Loss: {avg_loss:.4f}\n")

    # Save model
    path = f"./weights/{cfg.datasets.model_output}"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
main()