import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from transformers import RobertaTokenizer
from datascripts.speech_commands import SpeechCommandsText
from model.clap_trimodal import CLAPTriModal
import torch.nn.functional as F
import random
from dotenv import load_dotenv
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Config
load_dotenv()
access_token = os.getenv("HF_TOKEN")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print full config for logging

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and dataset
    print("Loading tokenizer and dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = SpeechCommandsText(tokenizer, split="test")
    idx2label = {v: k for k, v in test_dataset.group2idx.items()}
    labels_text = [f"This is the class: {label}" for label in test_dataset.group_labels]

    # Encode class descriptions
    with torch.no_grad():
        model = CLAPTriModal(cfg.model.audio_encoder, cfg.model.text_encoder, d_proj=cfg.model.d_proj, access_token=access_token).to(device)
        model.load_state_dict(torch.load(f"./weights/{cfg.datasets.model_output}", map_location=device))
        model.eval()

        class_tokens = tokenizer(labels_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
        class_embeds = model.class_text_encoder(class_tokens)  # (num_classes, d_proj)
        class_embeds = F.normalize(class_embeds, dim=-1)

    # Evaluation
    correct_audio, correct_text, correct_both = 0, 0, 0

    total = len(test_dataset)
    print("Running zero-shot inference on test set...")
    for i in range(total):
        waveform, text_inputs, label = test_dataset[i]
        waveform = waveform.unsqueeze(0).to(device)
        text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items()}
        label = torch.tensor([label]).to(device)

        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)  # (1, d_proj)
            z_text = model.input_text_encoder(text_inputs)  # (1, d_proj)

            # Mode 1: audio only
            sims_audio = torch.matmul(z_audio, class_embeds.T)
            pred_audio = torch.argmax(sims_audio, dim=1)
            correct_audio += (pred_audio == label).item()

            # Mode 2: text only
            sims_text = torch.matmul(z_text, class_embeds.T)
            pred_text = torch.argmax(sims_text, dim=1)
            correct_text += (pred_text == label).item()

            # Mode 3: average of both
            z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1)
            correct_both += (pred_both == label).item()

    print("\nZero-Shot Inference Results:")
    print(f"Total samples: {total}")
    print(f"Accuracy (Audio only):  {100 * correct_audio / total:.2f}%")
    print(f"Accuracy (Text only):   {100 * correct_text / total:.2f}%")
    print(f"Accuracy (Audio + Text): {100 * correct_both / total:.2f}%")


    # Sample some indices for testing
    test_indices = random.sample(range(len(test_dataset)), 5)
    test_samples = [test_dataset[i] for i in test_indices]

    # Inference modes
    for i, (waveform, text_inputs, label) in enumerate(test_samples):
        
        waveform = waveform.unsqueeze(0).to(device)
        text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items()}
        label = torch.tensor([label]).to(device)
        
        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)  # (1, d_proj)
            z_text = model.input_text_encoder(text_inputs)  # (1, d_proj)

            print(f"\nSample {i+1} | Ground truth class: {idx2label[label.item()]}")
            
            # Mode 1: Both modalities
            z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1).item()
            print(f"Prediction (audio + text): {idx2label[pred_both]}")

            # Mode 2: Audio only
            sims_audio = torch.matmul(z_audio, class_embeds.T)
            pred_audio = torch.argmax(sims_audio, dim=1).item()
            print(f"Prediction (audio only): {idx2label[pred_audio]}")

            # Mode 3: Text only
            sims_text = torch.matmul(z_text, class_embeds.T)
            pred_text = torch.argmax(sims_text, dim=1).item()
            print(f"Prediction (text only): {idx2label[pred_text]}")
        
        
main()