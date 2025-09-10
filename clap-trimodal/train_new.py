import os
import torch
import hydra
import wandb
import optuna
import gc

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch.nn.functional as F

from dotenv import load_dotenv
from evaluate_new import evaluate
from model.contrastive_loss import clip_contrastive_loss
from model_loader import load_class_embeds
from datascripts.dataset_loader import get_dataset, get_dataset_and_collate_fn
from checkpoint import save_checkpoint, load_checkpoint
from model.clap_trimodal import CLAPTriModal
from utils.optuna_log import get_optuna_filename, log_epoch_results
from train_utils import freeze_for_modality, build_optimizer_from_trainable
from datascripts.prompt_utils import get_prompt

from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


def train(cfg: DictConfig, return_val_metric: bool = False, trial: Optional[optuna.Trial] = None, trial_number: int = None, run_timestamp: str = None) -> Optional[float]:
    print(OmegaConf.to_yaml(cfg))
    
    run_id = None
    if cfg.dataset.get("model_checkpoint"):
        checkpoint_path = cfg.dataset.model_checkpoint
        print(f"Checking for W&B run ID in checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        run_id = checkpoint.get("wandb_id", None)
        
    
    
    wandb.init(
        project="clap-trimodal",
        name=cfg.get("run_name", None),
        id=run_id,
        resume="allow" if run_id else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print("W&B initialized")

    if trial is not None:
        log_path = get_optuna_filename(cfg.dataset.name, run_timestamp)

    print("Loading tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    print("Loading dataset...")
    # Load dataset and collate function
    train_dataset, collate_fn = get_dataset_and_collate_fn(cfg, tokenizer)
    val_dataset = get_dataset(cfg, tokenizer, "val")


    # 1) Build a single canonical vocab (prefer dataset-provided; else derive deterministically)
    if hasattr(train_dataset, "all_labels") and train_dataset.all_labels:
        label_vocab = list(train_dataset.all_labels)  # trust train split’s canonical order
    elif hasattr(val_dataset, "all_labels") and val_dataset.all_labels:
        label_vocab = list(val_dataset.all_labels)
    else:
        # fallback: collect from train labels (sorted for determinism)
        label_vocab = sorted(set(getattr(train_dataset, "labels", [])))

    # helpful checks
    assert all(isinstance(x, str) for x in label_vocab), "label_vocab must be list[str]"
    label_to_idx = {lbl: i for i, lbl in enumerate(label_vocab)}


    # Count class occurrences
    if hasattr(train_dataset, "labels") and train_dataset.labels:
        label_counts = Counter(train_dataset.labels)
        class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    drop_last = False if cfg.train.modality == "trimodal" else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=drop_last
    )

    label_names = getattr(val_dataset, "all_labels", None)

    print(f"Train set size: {len(train_dataset)} samples")
    print(f"Validation set size: {len(val_dataset)} samples")
    
    # Initialize model
    print("Initializing model...")
    dropout_rate = 0.5 if cfg.train.modality == "trimodal" else (0.2 if cfg.train.modality == "audio_text" else 0.1)
    model = CLAPTriModal(
        cfg.model.audio_encoder, cfg.model.text_encoder, d_proj=cfg.model.d_proj, access_token=access_token,
        init_tau=cfg.model.init_tau,
        min_logit_scale=cfg.model.min_logit_scale,
        max_logit_scale=cfg.model.max_logit_scale,
        dropout_rate=dropout_rate
    ).to(device)

    freeze_for_modality(model, cfg.train.modality)
    optimizer = build_optimizer_from_trainable(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # Optionally load model from checkpoint if provided in config
    if cfg.dataset.get("model_checkpoint"):
        model, optimizer, scheduler, start_epoch, best_val_acc = load_checkpoint(cfg, model, optimizer, scheduler, device)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.0
        print("No checkpoint specified. Training from scratch.")

    

    fixed_class_embeds = None
    if cfg.train.modality in ["text_only", "audio_only", "audio_text_unaligned"]:
        for p in model.class_text_encoder.parameters():
            p.requires_grad = False
        model.class_text_encoder.eval()

        with torch.no_grad():
            class_prompts = [get_prompt(lbl, cfg) for lbl in label_vocab]
            tokenized = tokenizer(
                class_prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=cfg.dataset.max_text_length
            ).to(device)
            fixed_class_embeds = model.class_text_encoder(tokenized)
            fixed_class_embeds = F.normalize(fixed_class_embeds, dim=-1)


    print("Starting training...")
    for epoch in range(start_epoch, cfg.train.epochs):
        print(f"\n------ Epoch {epoch} ------")

        model.train()
        total_loss, total_audio_text, total_audio_class, total_text_class = 0, 0, 0, 0

        start_step = 0
        start = time()
        
        for step, (audio, text_inputs, class_text_inputs, batch_labels) in enumerate(train_loader):

            if cfg.train.modality == "text_only":
                audio = None
                class_text_inputs = None
            elif cfg.train.modality == "audio_only":
                text_inputs = None
                class_text_inputs = None
            elif cfg.train.modality == "audio_text":
                class_text_inputs = None
            elif cfg.train.modality == "audio_text_unaligned":
                class_text_inputs = None
            # trimodal → keep all three
           
            audio = audio.to(device) if audio is not None else None
            text_inputs = ({k: v.to(device) for k, v in text_inputs.items()} if text_inputs is not None else None)
            class_text_inputs = ({k: v.to(device) for k, v in class_text_inputs.items()} if class_text_inputs is not None else None)

            
            out = model(audio=audio, input_text=text_inputs, class_text=class_text_inputs)

            audio_embed = out.get("audio_embed", None)
            text_embed  = out.get("input_text_embed", None)
            class_embed = out.get("class_text_embed", None)
            scale = out["contrastive_scale"]
            logit_scale_raw = out["logit_scale_raw"]
            
            
            # losses per modality
            loss_audio_text  = None
            loss_audio_class = None
            loss_text_class  = None
            losses, names = [], []

            targets = torch.tensor([label_to_idx[lbl] for lbl in batch_labels], device=device)

            if cfg.train.modality == "trimodal":
                # original behavior: CLIP for all three pairs
                # requires that you passed class_text_inputs so class_embed is not None
                loss_audio_text  = clip_contrastive_loss(audio_embed, text_embed,  scale, device=device)
                loss_audio_class = clip_contrastive_loss(audio_embed, class_embed, scale, device=device)
                loss_text_class  = clip_contrastive_loss(text_embed,  class_embed, scale, device=device)

                losses = [loss_audio_text, loss_audio_class, loss_text_class]
                names  = ["audio_text", "audio_class", "text_class"]

            elif cfg.train.modality == "audio_text":
                # CLIP just for A<->T
                loss_audio_text = clip_contrastive_loss(audio_embed, text_embed, scale, device=device)
                losses = [loss_audio_text]
                names  = ["audio_text"]

            elif cfg.train.modality == "audio_only":
                # classification vs fixed class anchors
                # fixed_class_embeds must be precomputed once before the loop (normalized), and
                # targets must be integer indices from labels
                logits_audio = (audio_embed @ fixed_class_embeds.T) * scale
                loss_audio_class = torch.nn.functional.cross_entropy(logits_audio, targets)

                losses = [loss_audio_class]
                names  = ["audio_class"]

            elif cfg.train.modality == "text_only":
                # classification vs fixed class anchors
                logits_text = (text_embed @ fixed_class_embeds.T) * scale
                loss_text_class = torch.nn.functional.cross_entropy(logits_text, targets)

                losses = [loss_text_class]
                names  = ["text_class"]

            elif cfg.train.modality == "audio_text_unaligned":
                # both encoders produce embeddings; classify each against fixed class anchors
                # audio branch
                loss_audio_class = None
                if audio_embed is not None:
                    logits_audio = (audio_embed @ fixed_class_embeds.T) * scale
                    loss_audio_class = torch.nn.functional.cross_entropy(logits_audio, targets)

                # text branch
                loss_text_class = None
                if text_embed is not None:
                    logits_text = (text_embed @ fixed_class_embeds.T) * scale
                    loss_text_class = torch.nn.functional.cross_entropy(logits_text, targets)

                # collect
                losses, names = [], []
                if loss_audio_class is not None:
                    losses.append(loss_audio_class); names.append("audio_class")
                if loss_text_class is not None:
                    losses.append(loss_text_class); names.append("text_class")

            # Weight losses if multiple
            if len(losses) > 1:
                sharpness = cfg.train.max_sharpness
                with torch.no_grad():
                    lv = torch.tensor([l.item() for l in losses], device=device)
                    weights = torch.softmax(lv / sharpness, dim=0)
                loss = sum(w * l for w, l in zip(weights, losses))
            else:
                loss = losses[0]
                weights = [1.0]


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if loss_audio_text  is not None: total_audio_text  += loss_audio_text.item()
            if loss_audio_class is not None: total_audio_class += loss_audio_class.item()
            if loss_text_class  is not None: total_text_class  += loss_text_class.item()

            
            # Logging
            wandb.log({f"loss/{n}": l.item() for n, l in zip(names, losses)})
            wandb.log({
                "loss/total": loss.item(),
                "logit_scale_raw": logit_scale_raw.item(),
                "contrastive_scale": scale.item(),
                "temperature": 1.0 / scale.item(),
            })

            if step % 10 == 0:
                line = " | ".join([f"{n}: {l.item():.4f}" for n, l in zip(names, losses)])
                print(f"\nStep {step:03d} | {line}")
                if len(losses) > 1:
                    print("Weights: " + " | ".join([f"{n}: {float(w):.4f}" for n, w in zip(names, weights)]))
                    print(f"Sharpness: {cfg.train.max_sharpness:.4f}")
                print(f"Temperature: {1.0 / scale.item():.4f} | Contrastive scale: {scale.item():.4f}")

            if start_step + 10 == step:
                    end = time()
                    print(f"Duration of 10 steps: {end-start}")
                    start_step = step
                    start = time()
                

        scheduler.step()
        
        denom = max(1, len(train_loader))
        avg_loss = total_loss / denom
        avg_audio_text_loss = total_audio_text / denom
        avg_audio_loss      = total_audio_class / denom
        avg_text_loss       = total_text_class / denom
        
        print("\n--- Evaluation ---")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Audio + Text: {avg_audio_text_loss:.4f} | Audio: {avg_audio_loss:.4f} | Text: {avg_text_loss:.4f}")
        
        acc_both, acc_audio, acc_text = evaluate(
            cfg, model, tokenizer, val_dataset,
            class_embeds=None,  # ignored
            emotion2idx={lbl: i for i, lbl in enumerate(label_vocab)},
            idx2emotion={i: lbl for i, lbl in enumerate(label_vocab)},
            device=device
        )

        if trial:
            log_epoch_results(log_path, trial_number, epoch, acc_both, acc_audio, acc_text)

        if cfg.train.modality == "trimodal" or cfg.train.modality == "audio_text":
            curr_best_acc = (acc_both + acc_audio + acc_text) / 3
        elif cfg.train.modality == "text_only":
            curr_best_acc = acc_text
        elif cfg.train.modality == "audio_only":
            curr_best_acc = acc_audio
        elif cfg.train.modality == "audio_text_unaligned":
            curr_best_acc = (acc_audio + acc_text) / 2.0
        else:
            curr_best_acc = float('-inf') 
        
        if trial:
            trial.report(curr_best_acc, step=epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        is_best = curr_best_acc > best_val_acc
        
        if is_best:
            best_val_acc = curr_best_acc
        
        if trial is None:
            if best_val_acc:
                save_checkpoint(
                    cfg, model, optimizer, scheduler, epoch, avg_loss, best_val_acc,
                    path=f"checkpoints/{cfg.dataset.model_output.lower()}",
                    is_best=is_best
                )
        
                    
        # Correct epoch-avg logging (don’t log last batch as avg)
        avg_metrics = {"avg_loss/avg_total": avg_loss}
        if cfg.train.modality in ["trimodal", "audio_text"]:
            avg_metrics["avg_loss/avg_audio_text"] = avg_audio_text_loss
        if cfg.train.modality in ["trimodal", "audio_text", "audio_only"]:
            avg_metrics["avg_loss/avg_audio_class"] = avg_audio_loss
        if cfg.train.modality in ["trimodal", "audio_text", "text_only"]:
            avg_metrics["avg_loss/avg_text_class"] = avg_text_loss
        wandb.log(avg_metrics)
        
        
        if cfg.train.modality in ["trimodal", "audio_text"]:
            wandb.log({
                "val/accuracy_audio_text": acc_both,
                "val/accuracy_audio": acc_audio,
                "val/accuracy_text": acc_text,
            })
        elif cfg.train.modality == "text_only":
            wandb.log({"val/accuracy_text": acc_text})
        #elif audio_only unchanged
        elif cfg.train.modality == "audio_only":
            wandb.log({"val/accuracy_audio": acc_audio})
        elif cfg.train.modality == "audio_text_unaligned":
            # no aligned "both" metric; log the two heads
            wandb.log({
                "val/accuracy_audio": acc_audio,
                "val/accuracy_text": acc_text,
            })

    wandb.finish()
    
    del model 
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect() 
    
    if return_val_metric:
        val_metric = {
            "best_val_acc": best_val_acc,
            "acc_audio_text": acc_both,
            "acc_text": acc_text,
            "acc_audio": acc_audio
        }
        return val_metric

    
    

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_train(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    run_train()