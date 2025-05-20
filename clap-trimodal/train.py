import os
import torch
import hydra
import wandb

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
from evaluation import evaluate
from model.contrastive_loss import clip_contrastive_loss
from model_loader import load_class_embeds
from datascripts.dataset_loader import get_dataset, get_dataset_and_collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint
from model.clap_trimodal import CLAPTriModal

import torch.cuda.amp as amp
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    run_id = None
    if cfg.dataset.get("model_checkpoint"):
        checkpoint_path = cfg.dataset.model_checkpoint
        print(f"Checking for W&B run ID in checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        run_id = checkpoint.get("wandb_id", None)
    
    
    wandb.init(
        project="clap-trimodal",
        name=cfg.get("run_name", None),
        id=run_id,
        resume="allow" if run_id else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print("W&B initialized")


    print("Loading tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    print("Loading dataset...")
    # Load dataset and collate function
    train_dataset, collate_fn = get_dataset_and_collate_fn(cfg, tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=16, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2,  # prefetch 2 batches per worker
        ) 

    val_dataset = get_dataset(cfg, tokenizer, "val")
    label_names = val_dataset.all_labels

    print(f"Train set size: {len(train_dataset)} samples")
    print(f"Validation set size: {len(val_dataset)} samples")
    
    # Initialize model
    print("Initializing model...")
    model = CLAPTriModal(
        cfg.model.audio_encoder, cfg.model.text_encoder, d_proj=cfg.model.d_proj, access_token=access_token
    ).to(device)
    
    optimizer = torch.optim.AdamW([
        {"params": model.audio_encoder.parameters(), "lr": cfg.train.lr_enc},
        {"params": model.input_text_encoder.parameters(), "lr": cfg.train.lr_enc},
        {"params": model.class_text_encoder.parameters(), "lr": cfg.train.lr_enc},
        {"params": [model.logit_scale], "lr": cfg.train.lr_proj},  
    ], lr=cfg.train.lr_proj)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    
    # scaler = amp.GradScaler()
    
    
    # Optionally load model from checkpoint if provided in config
    if cfg.dataset.get("model_checkpoint"):
        model, optimizer, scheduler, start_epoch, best_val_acc = load_checkpoint(cfg, model, optimizer, scheduler, device)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.0
        print("No checkpoint specified. Training from scratch.")


    print("Starting training...")
    for epoch in range(start_epoch, cfg.train.epochs):
        print(f"\n------ Epoch {epoch} ------\n")

        model.train()
        total_loss, total_audio_text, total_audio_class, total_text_class = 0, 0, 0, 0

        start_step = 0
        start = time()
        
        for step, (audio, text_inputs, class_text_inputs) in enumerate(train_loader):            
            audio = audio.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}
            
            
            # with amp.autocast():
            out = model(audio=audio, input_text=text_inputs, class_text=class_text_inputs)

            scale = out["contrastive_scale"]
            audio_embed = out["audio_embed"]
            text_embed = out["input_text_embed"]
            class_embed = out["class_text_embed"]

            # Contrastive losses between all 3 modalities (pairwise)
            loss_audio_text = clip_contrastive_loss(audio_embed, text_embed, scale, device=device)
            loss_audio_class = clip_contrastive_loss(audio_embed, class_embed, scale, device=device)
            loss_text_class = clip_contrastive_loss(text_embed, class_embed, scale, device=device)

            # progress = epoch / cfg.train.epochs
            # temperature = cfg.train.max_temp + (cfg.train.min_temp - cfg.train.max_temp) * progress
            # temperature = max(cfg.train.min_temp, temperature)  # Clamp to avoid undershooting
            temperature = cfg.train.max_temp

            # Proportional loss weights
            losses = [loss_audio_text, loss_audio_class, loss_text_class]
            loss_tensor = torch.tensor([l.item() for l in losses])
            weights = torch.softmax(loss_tensor / temperature, dim=0)
            loss = sum(w * l for w, l in zip(weights, losses))

            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_audio_text += loss_audio_text.item()
            total_audio_class += loss_audio_class.item()
            total_text_class += loss_text_class.item()
            
            wandb.log({
                "loss/total": loss.item(),
                "loss/audio_text": loss_audio_text.item(),
                "loss/audio_class": loss_audio_class.item(),
                "loss/text_class": loss_text_class.item(),
                "step": step
            })

            if step % 10 == 0:
                print(f"Step {step:03d}")
                print(
                    f"Loss: {loss.item():.4f} | A↔T: {loss_audio_text.item():.4f} | A↔C: {loss_audio_class.item():.4f} | T↔C: {loss_text_class.item():.4f}"
                )
                weight_audio_text, weight_aduio, weight_text = weights
                print(
                    f"Loss weights | A↔T: {weight_audio_text:.4f} | A↔C: {weight_aduio:.4f} | T↔C: {weight_text:.4f}"
                )
                print(F"Temperature: {temperature}")
            
            if start_step + 10 == step:
                    end = time()
                    print(f"Duration of 10 steps: {end-start}")
                    start_step = step
                    start = time()
                

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_audio_text_loss = total_audio_text / len(train_loader)
        avg_audio_loss = total_audio_class / len(train_loader)
        avg_text_loss = total_text_class / len(train_loader)
        
        print("\n--- Evaluation ---")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Audio + Text: {avg_audio_text_loss:.4f} | Audio: {avg_audio_loss:.4f} | Text: {avg_text_loss:.4f}")
        
        class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)
        acc_both, acc_audio, acc_text = evaluate(cfg, model, tokenizer, val_dataset, class_embeds, emotion2idx, idx2emotion, device)

        curr_best_acc = (acc_both + acc_audio + acc_text) / 3
        
        is_best = curr_best_acc > best_val_acc
        if is_best:
            best_val_acc = curr_best_acc

        save_checkpoint(
            model, optimizer, scheduler, epoch, avg_loss, best_val_acc,
            path=f"checkpoints/{cfg.dataset.model_output.lower()}",
            is_best=is_best
        )
        
                    
        wandb.log({
            "avg_loss/avg_total": loss.item(),
            "avg_loss/avg_audio_text": loss_audio_text.item(),
            "avg_loss/avg_audio_class": loss_audio_class.item(),
            "avg_loss/avg_text_class": loss_text_class.item(),
            "epoch": epoch,
        })

        wandb.log({
            "val/accuracy_audio_text": acc_both,
            "val/accuracy_audio": acc_audio,
            "val/accuracy_text": acc_text,
            "epoch": epoch,
        })
            

    wandb.finish()

if __name__ == "__main__":
    train()
