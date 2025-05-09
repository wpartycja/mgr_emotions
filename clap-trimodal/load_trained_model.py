from model.clap_trimodal import CLAPTriModal
from transformers import RobertaTokenizer
from datascripts.prompt_utils import get_prompt
import torch
import os
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("HF_TOKEN")

def load_trained_model(cfg, label_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    model = CLAPTriModal(
        cfg.model.audio_encoder,
        cfg.model.text_encoder,
        d_proj=cfg.model.d_proj,
        access_token=access_token
    ).to(device)

    model.load_state_dict(torch.load(f"./weights/{cfg.datasets.model_output}", map_location=device))
    model.eval()

    emotion2idx = {label: idx for idx, label in enumerate(label_names)}
    idx2emotion = {idx: label for label, idx in emotion2idx.items()}

    label_prompts = [get_prompt(label, cfg) for label in label_names]
    class_tokens = tokenizer(label_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
    class_embeds = model.class_text_encoder(class_tokens)
    class_embeds = torch.nn.functional.normalize(class_embeds, dim=-1)

    return model, tokenizer, class_embeds, emotion2idx, idx2emotion, device
