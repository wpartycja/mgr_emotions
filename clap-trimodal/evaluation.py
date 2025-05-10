import os
import torch
import hydra

from transformers import RobertaTokenizer, PreTrainedTokenizer, PreTrainedModel
from omegaconf import DictConfig
from typing import Dict
from torch import Tensor
from torch.utils.data import Dataset

from dotenv import load_dotenv
from model_loader import load_trained_model, load_class_embeds
from datascripts.dataset_loader import get_dataset
from datascripts.prompt_utils import get_prompt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


def print_class_descriptions(cfg: DictConfig, emotion2idx: Dict[str, int]) -> None:
    print("\nClass Descriptions:")
    for label, idx in emotion2idx.items():
        prompt = get_prompt(label, cfg)
        print(f"{label} ({idx}): {prompt}")


def evaluate(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    class_embeds: Tensor,
    emotion2idx: Dict[str, int],
    idx2emotion: Dict[int, str],
    device: torch.device,
) -> None:
    get_waveform = lambda x: x[0]
    get_label = lambda x: x[1]
    get_transcript = lambda x: x[2]

    correct_audio = correct_text = correct_both = 0
    total = len(test_dataset)

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

    print(f"Accuracy:")
    print(
        f"Audio + Text: {100 * correct_both / total:.2f}% | Audio only: {100 * correct_audio / total:.2f}% | Text only: {100 * correct_text / total:.2f}%  "
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_evaluation(cfg: DictConfig) -> None:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = get_dataset(cfg, tokenizer, "test")
    label_names = test_dataset.all_labels
    model, tokenizer, device = load_trained_model(cfg)
    class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)

    print_class_descriptions(cfg, emotion2idx)
    print(f"\nEvaluating on {len(test_dataset)} samples from {cfg.datasets.name.lower()}...")
    evaluate(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device)


if __name__ == "__main__":
    run_evaluation()
