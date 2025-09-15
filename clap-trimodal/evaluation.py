import os
import torch
import hydra

from transformers import RobertaTokenizer, PreTrainedTokenizer, PreTrainedModel
from omegaconf import DictConfig
from typing import Dict, List, Union
from torch import Tensor
from torch.utils.data import Dataset
from datetime import datetime

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from dotenv import load_dotenv
from model_loader import load_trained_model, load_class_embeds
from datascripts.dataset_loader import get_dataset
from datascripts.prompt_utils import get_prompt
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")

IMG_SAVE_DIR = 'png'


def print_class_descriptions(cfg: DictConfig, emotion2idx: Dict[str, int]) -> None:
    print("\nClass Descriptions:")
    for label, idx in emotion2idx.items():
        prompt = get_prompt(label, cfg)
        print(f"{label} ({idx}): {prompt}")


def compute_metrics(y_true, y_pred):
    wa = accuracy_score(y_true, y_pred) * 100  
    ua = balanced_accuracy_score(y_true, y_pred) * 100  
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        "weighted_accuracy": wa,
        "unweighted_accuracy": ua,
        "precision": precision * 100,
        "f1": f1 * 100,
    }


def plot_confusion_matrix(y_true, y_pred, labels, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.round(cm_normalized * 100, 1)  # percentages with 1 decimal
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(cmap="RdPu", xticks_rotation=45, values_format=".1f")  # show decimals
    plt.title(title + " (in %)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_metrics(metrics_audio, metrics_text, metrics_both):
    def print_line(label, key):
        print(f"{label}:")
        audio_text = metrics_both[key] if metrics_both is not None else float("nan")
        audio = metrics_audio[key] if metrics_audio is not None else float("nan")
        text = metrics_text[key] if metrics_text is not None else float("nan")

        print(
            f"Audio + Text: {audio_text:.2f}% | "
            f"Audio only: {audio:.2f}% | "
            f"Text only: {text:.2f}%\n"
        )

    print_line("Weighted Accuracy (WA)", "weighted_accuracy")
    print_line("Unweighted Accuracy (UA)", "unweighted_accuracy")
    print_line("Macro Precision", "precision")
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

    if class_embeds is None:
        num_classes = len(emotion2idx)
        label_vocab = [None] * num_classes
        for lbl, i in emotion2idx.items():
            label_vocab[i] = lbl

        prompts = [get_prompt(lbl, cfg) for lbl in label_vocab]
        with torch.inference_mode():
            tok = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=cfg.dataset.max_text_length
            ).to(device)
            class_embeds = model.class_text_encoder(tok)   # (K, d), already normalized by your encoder
            class_embeds = F.normalize(class_embeds, dim=-1)

    else:
        class_embeds = F.normalize(class_embeds.to(device), dim=-1)

    use_audio = cfg.train.modality in ["trimodal", "audio_text", "audio_only", "audio_text_unaligned"]
    use_text  = cfg.train.modality in ["trimodal", "audio_text", "text_only", "audio_text_unaligned"]
    combine_modalities = cfg.train.modality in ["trimodal", "audio_text"]
    use_both = (use_audio and use_text and combine_modalities)

    y_true, y_pred_audio, y_pred_text, y_pred_both = [], [], [], []

    model.eval()
    with torch.inference_mode():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            label_str = get_label(sample)
            y_true.append(emotion2idx[label_str])

            z_audio = None
            z_text  = None

            if use_audio:
                waveform = get_waveform(sample).unsqueeze(0).to(device)
                z_audio = model.audio_encoder(waveform)  # normalized by your model

            if use_text:
                transcript = get_transcript(sample)
                text_inputs = tokenizer(
                    transcript,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.dataset.max_text_length,
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                z_text = model.input_text_encoder(text_inputs)  # normalized

            if use_audio:
                sims_audio = torch.matmul(z_audio, class_embeds.T)
                pred_audio = torch.argmax(sims_audio, dim=1).item()
                y_pred_audio.append(pred_audio)

            if use_text:
                sims_text = torch.matmul(z_text, class_embeds.T)
                pred_text = torch.argmax(sims_text, dim=1).item()
                y_pred_text.append(pred_text)

            if use_both:
                z_avg = F.normalize((z_audio + z_text) / 2, dim=-1)
                sims_both = torch.matmul(z_avg, class_embeds.T)
                pred_both = torch.argmax(sims_both, dim=1).item()
                y_pred_both.append(pred_both)

    if extended_metrics:
        metrics_audio = compute_metrics(y_true, y_pred_audio) if use_audio else None
        metrics_text  = compute_metrics(y_true, y_pred_text)  if use_text  else None
        metrics_both  = compute_metrics(y_true, y_pred_both)  if use_both  else None

        def _p(m, key): return f"{m[key]:.2f}%" if m else "N/A"

        if use_both:
            print_metrics(metrics_audio, metrics_text, metrics_both)
        elif use_text and use_audio:
            print_metrics(metrics_audio, metrics_text, None)
        elif use_text and not use_audio:
            print_metrics(None, metrics_text, None)
        elif use_audio and not use_text:
            print_metrics(metrics_audio, None, None)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels_order = [lbl for lbl, _i in sorted(emotion2idx.items(), key=lambda x: x[1])]

        os.makedirs(IMG_SAVE_DIR, exist_ok=True)

        save_name = cfg.dataset.model_checkpoint.split('/')[1][:-2]

        if use_audio:
            plot_confusion_matrix(y_true, y_pred_audio, labels_order,
                                "Audio only", f"{IMG_SAVE_DIR}/{timestamp}_{save_name}_{cfg.train.modality}_confusion_audio_{cfg.dataset.name}.png")
        if use_text:
            plot_confusion_matrix(y_true, y_pred_text, labels_order,
                                "Text only", f"{IMG_SAVE_DIR}/{timestamp}_{save_name}_{cfg.train.modality}_confusion_text_{cfg.dataset.name}.png")
        if use_both:
            plot_confusion_matrix(y_true, y_pred_both, labels_order,
                                "Audio + Text", f"{IMG_SAVE_DIR}/{timestamp}_{save_name}_{cfg.train.modality}_confusion_both_{cfg.dataset.name}.png")

        return {"audio": metrics_audio, "text": metrics_text, "both": metrics_both if use_both else None}
    else:
        acc_both  = balanced_accuracy_score(y_true, y_pred_both) * 100 if use_both else float("nan")
        acc_audio = balanced_accuracy_score(y_true, y_pred_audio) * 100 if use_audio else float("nan")
        acc_text  = balanced_accuracy_score(y_true, y_pred_text)  * 100 if use_text  else float("nan")

        if use_both:
            print(f"Balanced Accuracy: Audio + Text: {acc_both:.2f}% | Audio only: "
                f"{(acc_audio if use_audio else float('nan')):.2f}% | Text only: "
                f"{(acc_text if use_text else float('nan')):.2f}%\n")
            return [acc_both, acc_audio, acc_text]
        elif use_text and use_audio:
            # unaligned dual-head (audio_text_unaligned): no fused score
            print(f"Balanced Accuracy: Audio only: {acc_audio:.2f}% | Text only: {acc_text:.2f}%\n")
            return [float("nan"), acc_audio, acc_text]
        elif use_text and not use_audio:
            print(f"Balanced Accuracy: Text only: {acc_text:.2f}%\n")
            return [float("nan"), float("nan"), acc_text]
        else:  # audio-only
            print(f"Balanced Accuracy: Audio only: {acc_audio:.2f}%\n")
            return [float("nan"), acc_audio, float("nan")]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_evaluation(cfg: DictConfig) -> None:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    test_dataset = get_dataset(cfg, tokenizer, "test")

    if hasattr(test_dataset, "all_labels") and test_dataset.all_labels:
        label_vocab = list(test_dataset.all_labels)
    else:
        label_vocab = sorted(set(getattr(test_dataset, "labels", [])))

    emotion2idx = {lbl: i for i, lbl in enumerate(label_vocab)}
    idx2emotion = {i: lbl for i, lbl in enumerate(label_vocab)}

    model, tokenizer, device = load_trained_model(cfg)

    print_class_descriptions(cfg, emotion2idx)
    print(f"\nEvaluating on {len(test_dataset)} samples from {cfg.dataset.name.lower()}...")

    evaluate(cfg, model, tokenizer, test_dataset, None, emotion2idx, idx2emotion, device, True)


if __name__ == "__main__":
    run_evaluation()
