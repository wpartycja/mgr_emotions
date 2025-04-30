import torch
import torch.nn as nn
import torch.nn.functional as F
from model.audio_encoder import get_audio_encoder
from model.text_encoder import get_text_encoder


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        return self.layer_norm(embed1 + embed2)

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name: str, access_token: str, d_out: int, ):
        super().__init__()
        self.base = get_audio_encoder(audioenc_name)(access_token)
        d_in = self.base.output_dim
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        features = out_dict['embedding']
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, textenc_name: str, access_token: str, d_out: int, transformer_embed_dim: int = 768):
        super().__init__()
        self.base = get_text_encoder(textenc_name, access_token)
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        output = self.base(**x)
        cls_token = output.last_hidden_state[:, 0, :]
        projected = self.projection(cls_token)
        return F.normalize(projected, dim=-1)

class CLAPTriModal(nn.Module):
    def __init__(self,
                 audioenc_name: str,
                 textenc_name: str,
                 d_proj: int,
                 access_token: str):
        super().__init__()
        self.audio_encoder = AudioEncoder(audioenc_name, access_token, d_proj)
        self.input_text_encoder = TextEncoder(textenc_name, access_token, d_proj)
        self.class_text_encoder = TextEncoder(textenc_name, access_token, d_proj)

        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.min_temp = 0.01
        self.max_temp = 0.5

    def forward(self, audio=None, input_text=None, class_text=None):
        audio_embed = self.audio_encoder(audio) if audio is not None else None
        input_text_embed = self.input_text_encoder(input_text) if input_text is not None else None
        class_text_embed = self.class_text_encoder(class_text) if class_text is not None else None

        contrastive_scale = torch.clamp(self.logit_scale.exp(), self.min_temp, self.max_temp)

        return {
            'audio_embed': audio_embed,
            'input_text_embed': input_text_embed,
            'class_text_embed': class_text_embed,
            'contrastive_scale': contrastive_scale
        }
