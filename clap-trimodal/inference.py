import os
import random
import torch
import hydra

from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
from model_loader import load_trained_model, load_class_embeds
from datascripts.dataset_loader import get_dataset


load_dotenv()
access_token = os.getenv("HF_TOKEN")


def inference(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device):
    get_waveform = lambda x: x[0]
    get_label = lambda x: x[1]
    get_transcript = lambda x: x[2]

    print("Inspecting 5 random samples...")
    test_indices = random.sample(range(len(test_dataset)), 5)
    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        label = get_label(sample)
        waveform = get_waveform(sample).unsqueeze(0).to(device)
        transcript = get_transcript(sample)

        text_inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=64)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)
            z_text = model.input_text_encoder(text_inputs)

            print(f"\nSample {i+1} | Transcript: '{transcript}' | Ground truth: {label}")

            z_avg = torch.nn.functional.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1).item()
            print(f"Prediction (audio + text): {idx2emotion[pred_both]}")

            sims_audio = torch.matmul(z_audio, class_embeds.T)
            pred_audio = torch.argmax(sims_audio, dim=1).item()
            print(f"Prediction (audio only): {idx2emotion[pred_audio]}")

            sims_text = torch.matmul(z_text, class_embeds.T)
            pred_text = torch.argmax(sims_text, dim=1).item()
            print(f"Prediction (text only): {idx2emotion[pred_text]}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_inference(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = get_dataset(cfg, tokenizer, "test")
    label_names = test_dataset.all_labels
    model, tokenizer, device = load_trained_model(cfg)
    class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)

    inference(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device)


if __name__ == "__main__":
    run_inference()
