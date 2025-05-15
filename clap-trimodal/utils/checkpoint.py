import torch
import wandb
import os


def load_checkpoint(cfg, model, optimizer, scheduler, device):
    model_path = cfg.dataset.model_checkpoint
    print(f"Loading model from checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_acc = checkpoint.get("best_val_acc")
    
    print(f"Resumed from epoch {start_epoch}, loss {checkpoint.get('loss', 'N/A')}")
    
    return model, optimizer, scheduler, start_epoch, best_val_acc


def save_checkpoint(
    model, optimizer, scheduler, epoch: int, loss: float, best_val_acc: float, path: str, is_best: bool = False, use_wandb: bool = True
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), 
        "epoch": epoch,
        "loss": loss,
        "best_val_acc": best_val_acc
    }
    
    path_w_epoch = f"{path.split('.')[0]}_{epoch}.pt"
    torch.save(checkpoint, path_w_epoch)
    print(f"Saved checkpoint to {path_w_epoch}")

    if is_best:
        best_path = path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path}")
        wandb.save(best_path)