# Zero-shot inference using tri-modal CLAP on RAVDESS (audio, text, or both vs. class descriptions)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn.functional as F
import random
from dotenv import load_dotenv
from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra

from model.clap_trimodal import CLAPTriModal
from datascripts.prompt_utils import get_prompt
from datascripts.loader import load_dataset

# Load .env for Hugging Face token
load_dotenv()
access_token = os.getenv("HF_TOKEN")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print config for logging

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    print("\n🔁 Loading tokenizer and dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    # Load dataset and accessors
    test_dataset = load_dataset(cfg, tokenizer)
    get_waveform = lambda x: x[0]
    get_label = lambda x: x[1]
    get_transcript = lambda x: x[2]
    all_labels = test_dataset.all_labels
    
    emotion2idx = {emotion: idx for idx, emotion in enumerate(all_labels)}
    idx2emotion = {idx: emotion for emotion, idx in emotion2idx.items()}
    
    # Generate class descriptions from cfg (either template or per-label)
    labels_text = [get_prompt(label, cfg) for label in all_labels]

    with torch.no_grad():
        model = CLAPTriModal(
            cfg.model.audio_encoder,
            cfg.model.text_encoder,
            d_proj=cfg.model.d_proj,
            access_token=access_token
        ).to(device)

        model.load_state_dict(torch.load(f"./weights/{cfg.datasets.model_output}", map_location=device))
        model.eval()

        class_tokens = tokenizer(labels_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
        class_embeds = model.class_text_encoder(class_tokens)
        class_embeds = F.normalize(class_embeds, dim=-1)

    correct_audio = correct_text = correct_both = 0
    total = len(test_dataset)
    print(f"\n🔍 Running zero-shot inference on {total} samples from {cfg.datasets.name.lower()}...")

    for i in range(total):
        sample = test_dataset[i]
        label = get_label(sample)
        label_idx = torch.tensor([emotion2idx[label]]).to(device)

        waveform = get_waveform(sample).unsqueeze(0).to(device)
        transcript = get_transcript(sample)

        text_inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=64)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)
            z_text = model.input_text_encoder(text_inputs)

            sims_audio = torch.matmul(z_audio, class_embeds.T)
            pred_audio = torch.argmax(sims_audio, dim=1)
            correct_audio += (pred_audio == label_idx).item()

            sims_text = torch.matmul(z_text, class_embeds.T)
            pred_text = torch.argmax(sims_text, dim=1)
            correct_text += (pred_text == label_idx).item()

            z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1)
            correct_both += (pred_both == label_idx).item()

    print("\n🎯 Zero-Shot Inference Results:")
    print(f"Total samples: {total}")
    print(f"Accuracy (Audio only):  {100 * correct_audio / total:.2f}%")
    print(f"Accuracy (Text only):   {100 * correct_text / total:.2f}%")
    print(f"Accuracy (Audio + Text): {100 * correct_both / total:.2f}%")

    # Random 5 samples for inspection
    print("\n🔍 Inspecting 5 random samples...")
    test_indices = random.sample(range(len(test_dataset)), 5)
    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        label = get_label(sample)
        label_idx = emotion2idx[label]
        waveform = get_waveform(sample).unsqueeze(0).to(device)
        transcript = get_transcript(sample)

        text_inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=64)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)
            z_text = model.input_text_encoder(text_inputs)

            print(f"\nSample {i+1} | Transcript: '{transcript}' | Ground truth: {label}")

            z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1).item()
            print(f"Prediction (audio + text): {idx2emotion[pred_both]}")

            sims_audio = torch.matmul(z_audio, class_embeds.T)
            pred_audio = torch.argmax(sims_audio, dim=1).item()
            print(f"Prediction (audio only): {idx2emotion[pred_audio]}")

            sims_text = torch.matmul(z_text, class_embeds.T)
            pred_text = torch.argmax(sims_text, dim=1).item()
            print(f"Prediction (text only): {idx2emotion[pred_text]}")


if __name__ == "__main__":
    main()
