import os
import torch
import hydra

from transformers import RobertaTokenizer, PreTrainedTokenizer, PreTrainedModel
from omegaconf import DictConfig
from typing import Dict, List, Union
from torch import Tensor
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
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


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


def print_metrics(metrics_audio, metrics_text, metrics_both):
    def print_line(label, key):
        print(f"{label}:")
        print(
            f"Audio + Text: {metrics_both[key]:.2f}% | "
            f"Audio only: {metrics_audio[key]:.2f}% | "
            f"Text only: {metrics_text[key]:.2f}%\n"
        )

    print_line("Accuracy", "accuracy")
    print_line("Balanced Accuracy", "balanced_accuracy")
    print_line("Macro Precision", "precision")
    print_line("Macro Recall", "recall")
    print_line("Macro F1-score", "f1")

def evaluate(
    cfg: DictConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    class_embeds: Tensor,
    emotion2idx: Dict[str, int],
    idx2emotion: Dict[int, str],
    device: torch.device,
    extended_metrics: bool = False,
) -> Union[List[float], Dict[str, Dict[str, float]]]:
    get_waveform = lambda x: x[0]
    get_label = lambda x: x[1]
    get_transcript = lambda x: x[2]

    y_true = []
    y_pred_audio = []
    y_pred_text = []
    y_pred_both = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        label = get_label(sample)
        label_idx = torch.tensor([emotion2idx[label]]).to(device)
        y_true.append(label_idx.item())

        waveform = get_waveform(sample).unsqueeze(0).to(device)
        transcript = get_transcript(sample)

        text_inputs = tokenizer(
            transcript,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            z_audio = model.audio_encoder(waveform)
            z_text = model.input_text_encoder(text_inputs)
            z_avg = torch.nn.functional.normalize((z_audio + z_text) / 2, dim=-1)
            
            sims_audio = torch.matmul(z_audio, class_embeds.T)
            sims_text = torch.matmul(z_text, class_embeds.T)
            sims_both = torch.matmul(z_avg, class_embeds.T)

            pred_audio = torch.argmax(sims_audio, dim=1).item()
            pred_text = torch.argmax(sims_text, dim=1).item()
            pred_both = torch.argmax(sims_both, dim=1).item()

            y_pred_audio.append(pred_audio)
            y_pred_text.append(pred_text)
            y_pred_both.append(pred_both)

    if extended_metrics:
        metrics_audio = compute_metrics(y_true, y_pred_audio)
        metrics_text = compute_metrics(y_true, y_pred_text)
        metrics_both = compute_metrics(y_true, y_pred_both)

        print_metrics(metrics_audio, metrics_text, metrics_both)

        return {
            "audio": metrics_audio,
            "text": metrics_text,
            "both": metrics_both,
        }
    else:
        acc_audio = balanced_accuracy_score(y_true, y_pred_audio) * 100
        acc_text = balanced_accuracy_score(y_true, y_pred_text) * 100
        acc_both = balanced_accuracy_score(y_true, y_pred_both) * 100

        print("Balanced Accuracy:")
        print(f"Audio + Text: {acc_both:.2f}% | Audio only: {acc_audio:.2f}% | Text only: {acc_text:.2f}%\n")

        return [acc_both, acc_audio, acc_text]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_evaluation(cfg: DictConfig) -> None:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = get_dataset(cfg, tokenizer, "test")
    label_names = test_dataset.all_labels
    model, tokenizer, device = load_trained_model(cfg)
    class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)

    print_class_descriptions(cfg, emotion2idx)
    print(f"\nEvaluating on {len(test_dataset)} samples from {cfg.dataset.name.lower()}...")
    evaluate(cfg, model, tokenizer, test_dataset, class_embeds, emotion2idx, idx2emotion, device, True)


if __name__ == "__main__":
    run_evaluation()
