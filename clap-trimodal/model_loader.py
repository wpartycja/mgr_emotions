import os
import torch

from transformers import RobertaTokenizer, PreTrainedTokenizer
from omegaconf import DictConfig
from typing import Tuple, Dict, List
from torch import Tensor

from model.clap_trimodal import CLAPTriModal
from datascripts.prompt_utils import get_prompt
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


def load_trained_model(cfg: DictConfig) -> Tuple[CLAPTriModal, PreTrainedTokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    model = CLAPTriModal(
        cfg.model.audio_encoder, cfg.model.text_encoder, d_proj=cfg.model.d_proj, access_token=access_token
    ).to(device)

    model.load_state_dict(torch.load(f"./weights/{cfg.datasets.model_output}", map_location=device))
    model.eval()

    return model, tokenizer, device


def load_class_embeds(
    cfg: DictConfig, model: CLAPTriModal, tokenizer: PreTrainedTokenizer, label_names: List[str], device: torch.device
) -> Tuple[Tensor, Dict[str, int], Dict[int, str]]:

    emotion2idx = {label: idx for idx, label in enumerate(label_names)}
    idx2emotion = {idx: label for label, idx in emotion2idx.items()}

    label_prompts = [get_prompt(label, cfg) for label in label_names]
    class_tokens = tokenizer(label_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
    class_embeds = model.class_text_encoder(class_tokens)
    class_embeds = torch.nn.functional.normalize(class_embeds, dim=-1)

    return class_embeds, emotion2idx, idx2emotion
