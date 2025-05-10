import torch
import wandb
import os

def save_checkpoint(
    model, optimizer, epoch: int, loss: float, path: str, is_best: bool = False, use_wandb: bool = True
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    if is_best:
        best_path = path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Best model updated: {best_path}")
        wandb.save(best_path)