import torch
import wandb
import os
from train_utils import build_optimizer_from_trainable

def load_checkpoint(cfg, model, optimizer, scheduler, device):
    model_path = cfg.dataset.model_checkpoint
    print(f"Loading model from checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    same_context = (checkpoint['modality'] == cfg.train.modality) and (checkpoint['dataset'] == cfg.dataset.name)

    start_epoch = checkpoint.get("epoch", 0)
    best_val_acc = checkpoint.get("best_val_acc")


    if same_context:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Resumed optimizer and scheduler (modality={checkpoint['modality']}, dataset={checkpoint['dataset']}) "
                f"from epoch {start_epoch}, loss {checkpoint['loss']}")
    else:
        print(f"Context changed,\nPrevious modality: {checkpoint['modality']} changed for {cfg.train.modality}\n" \
              f"Previous dataset: {checkpoint['dataset']} changed for {cfg.dataset.name}\n" \
               "Rebuilding new optimizer and scheduler")
        optimizer = build_optimizer_from_trainable(model, cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
        
    
    return model, optimizer, scheduler, start_epoch, best_val_acc


def save_checkpoint(cfg, 
    model, optimizer, scheduler, epoch: int, loss: float, best_val_acc: float, path: str, is_best: bool = False
):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), 
        "epoch": epoch,
        "loss": loss,
        "best_val_acc": best_val_acc,
        "wandb_id": wandb.run.id,
        "modality": cfg.train.modality,
        "dataset": cfg.dataset.name,
    }
    
    path_w_epoch = f"{os.path.splitext(path)[0]}_{epoch}.pt" 
    torch.save(checkpoint, path_w_epoch)
    print(f"Saved checkpoint to {path_w_epoch}")

    if is_best:
        best_path = path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path}")
        wandb.save(best_path)
    