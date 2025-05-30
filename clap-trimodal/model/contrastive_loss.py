import torch
import torch.nn.functional as F

from torch import Tensor


def clip_contrastive_loss(x_embed: Tensor, y_embed: Tensor, scale: float, device: torch.device) -> Tensor:
    if x_embed is None or y_embed is None:
        return torch.tensor(0.0, device=device)

    batch_size = min(x_embed.size(0), y_embed.size(0))
    x_embed = x_embed[:batch_size]
    y_embed = y_embed[:batch_size]

    # Normalize
    x_embed = F.normalize(x_embed, dim=-1)
    y_embed = F.normalize(y_embed, dim=-1)

    # Similarity
    sim = torch.matmul(x_embed, y_embed.T) * scale

    # Targets = diagonals
    targets = torch.arange(batch_size, device=device)

    loss_x = F.cross_entropy(sim, targets)
    loss_y = F.cross_entropy(sim.T, targets)

    return (loss_x + loss_y) / 2
