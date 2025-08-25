import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Dict

from model.audio_encoder import get_audio_encoder
from model.text_encoder import get_text_encoder


class Projection(nn.Module):
    """Two-layer projection block with GELU, residual connection, dropout, and layer normalization."""

    def __init__(self, d_in: int, d_out: int, p: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        return self.layer_norm(embed1 + embed2)


class AudioEncoder(nn.Module):
    """Wrapper for audio encoder with projection and normalization."""

    def __init__(
        self,
        audioenc_name: str,
        access_token: str,
        d_out: int,
    ):
        super().__init__()
        self.base = get_audio_encoder(audioenc_name)(access_token)
        d_in = self.base.output_dim
        self.projection = Projection(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        out_dict = self.base(x)
        features = out_dict["embedding"]
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)


class TextEncoder(nn.Module):
    """Wrapper for text transformer with projection and normalization."""

    def __init__(self, textenc_name: str, access_token: str, d_out: int, transformer_embed_dim: int = 768):
        super().__init__()
        self.base = get_text_encoder(textenc_name, access_token)
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(**x)
        cls_token = output.last_hidden_state[:, 0, :]
        projected = self.projection(cls_token)
        return F.normalize(projected, dim=-1)


class CLAPTriModal(nn.Module):
    """Tri-modal CLAP model supporting audio ↔ input text ↔ class text embeddings."""

    def __init__(self, audioenc_name: str, textenc_name: str, d_proj: int, access_token: str, init_tau: float, min_logit_scale: float, max_logit_scale: float,):
        super().__init__()
        self.audio_encoder = AudioEncoder(audioenc_name, access_token, d_proj)
        self.input_text_encoder = TextEncoder(textenc_name, access_token, d_proj)
        self.class_text_encoder = TextEncoder(textenc_name, access_token, d_proj)

        self.init_tau = init_tau
        self.logit_scale = nn.Parameter(torch.tensor(1 / self.init_tau).log())
        self.min_logit_scale = min_logit_scale
        self.max_logit_scale = max_logit_scale

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        input_text: Optional[Dict[str, torch.Tensor]] = None,
        class_text: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        audio_embed = self.audio_encoder(audio) if audio is not None else None
        input_text_embed = self.input_text_encoder(input_text) if input_text is not None else None
        class_text_embed = self.class_text_encoder(class_text) if class_text is not None else None

        contrastive_scale = torch.clamp(self.logit_scale.exp(), self.min_logit_scale, self.max_logit_scale)

        return {
            "audio_embed": audio_embed,
            "input_text_embed": input_text_embed,
            "class_text_embed": class_text_embed,
            "contrastive_scale": contrastive_scale,
            "logit_scale_raw": self.logit_scale, 
        }
