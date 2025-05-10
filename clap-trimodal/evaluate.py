import os
import torch
from dotenv import load_dotenv
from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra

from load_trained_model import load_trained_model, load_class_embeds
from datascripts.loader import get_dataset
from datascripts.prompt_utils import get_prompt

load_dotenv()
access_token = os.getenv("HF_TOKEN")


def evaluate(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device):
    get_waveform = lambda x: x[0]
    get_label = lambda x: x[1]
    get_transcript = lambda x: x[2]

    print("\nClass Descriptions:")
    for label, idx in emotion2idx.items():
        prompt = get_prompt(label, cfg)
        print(f"{label} ({idx}): {prompt}")

    correct_audio = correct_text = correct_both = 0
    total = len(test_dataset)
    print(f"\nRunning zero-shot inference on {total} samples from {cfg.datasets.name.lower()}...")

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

            z_avg = torch.nn.functional.normalize((z_audio + z_text) / 2, dim=-1)
            sims_both = torch.matmul(z_avg, class_embeds.T)
            pred_both = torch.argmax(sims_both, dim=1)
            correct_both += (pred_both == label_idx).item()

    print("\n🎯 Zero-Shot Inference Results:")
    print(f"Total samples: {total}")
    print(f"Accuracy (Audio only):  {100 * correct_audio / total:.2f}%")
    print(f"Accuracy (Text only):   {100 * correct_text / total:.2f}%")
    print(f"Accuracy (Audio + Text): {100 * correct_both / total:.2f}%")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = get_dataset(cfg, tokenizer, "test")
    label_names = test_dataset.all_labels
    model, tokenizer, device = load_trained_model(cfg)
    class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)

    evaluate(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device)


main()