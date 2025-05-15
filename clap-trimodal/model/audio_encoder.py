import torch
import torch.nn as nn

from typing import Type
from transformers import Wav2Vec2Model, WavLMModel, HubertModel, AutoModel


def get_audio_encoder(name: str) -> Type[nn.Module]:
    if name == "Wav2Vec":
        return Wav2Vec
    elif name == "WavLM":
        return WavLM
    elif name == "HuBERT":
        return HuBERT
    elif name == "DistilHuBERT":
        return DistilHuBERT
    else:
        raise Exception(f"The audio encoder name {name} is incorrect or not supported")


class Wav2Vec(nn.Module):
    """Wrapper for facebook/wav2vec2 model that outputs pooled embeddings."""

    def __init__(self, access_token: str, model_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name, token=access_token)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_waveform, attention_mask=None):
        outputs = self.model(input_waveform, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return {"embedding": pooled}


class DistilHuBERT(nn.Module):
    """Wrapper for ntu-spml/distilhubert model that outputs pooled embeddings."""

    def __init__(self, access_token: str, model_name: str = "ntu-spml/distilhubert"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, token=access_token)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_waveform, attention_mask=None):
        outputs = self.model(input_waveform, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return {"embedding": pooled}


class WavLM(nn.Module):
    """Wrapper for microsoft/wavlm-base model that outputs pooled embeddings."""

    def __init__(self, model_name: str = "microsoft/wavlm-base"):
        super().__init__()
        self.model = WavLMModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_waveform, attention_mask=None):
        outputs = self.model(input_waveform, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return {"embedding": pooled}


class HuBERT(nn.Module):
    """Wrapper for facebook/hubert-base-ls960 model that outputs pooled embeddings."""

    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        super().__init__()
        self.model = HubertModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_waveform, attention_mask=None):
        outputs = self.model(input_waveform, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return {"embedding": pooled}
