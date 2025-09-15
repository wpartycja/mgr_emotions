# calibrate_ece.py
import os
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from dotenv import load_dotenv
import torch.nn.functional as F
from hydra.utils import get_original_cwd

from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List, Tuple

from datascripts.dataset_loader import get_dataset
from datascripts.prompt_utils import get_prompt
from model_loader import load_trained_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def compute_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    min_bin_count_mce: int = 5,
    n_resamples_ace: int = 20,
    scale: float = 1.0,
):
    """
    Compute calibration metrics: ECE, MCE (raw & filtered), SCE, ACE,
    plus fixed-bin and adaptive-bin stats for plotting.
    """

    N, K = logits.shape
    L = logits * scale
    L = L - L.max(axis=1, keepdims=True)  # stability
    P = np.exp(L); P /= P.sum(axis=1, keepdims=True)

    conf = np.max(P, axis=1)
    pred = np.argmax(P, axis=1)
    corr = (pred == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_cnt = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            bin_cnt[b] = mask.sum()
            bin_acc[b] = corr[mask].mean()
            bin_conf[b] = conf[mask].mean()

    gaps = np.abs(bin_acc - bin_conf)
    weights = bin_cnt / N
    ece = float(np.sum(weights * gaps))
    mce_raw = float(np.max(gaps)) if np.any(bin_cnt) else 0.0
    valid_bins = bin_cnt >= min_bin_count_mce
    mce_filtered = float(np.max(gaps[valid_bins]) if np.any(valid_bins) else 0.0)

    stats_fixed = (bin_edges, bin_acc, bin_conf, bin_cnt)

    q_edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    q_edges = np.unique(q_edges)

    bin_acc_a, bin_conf_a, bin_cnt_a = [], [], []
    for b in range(len(q_edges) - 1):
        lo, hi = q_edges[b], q_edges[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            bin_cnt_a.append(mask.sum())
            bin_acc_a.append(corr[mask].mean())
            bin_conf_a.append(conf[mask].mean())

    stats_adaptive = (q_edges, np.array(bin_acc_a), np.array(bin_conf_a), np.array(bin_cnt_a))

    sce_terms = []
    for k in range(K):
        conf_k = P[:, k]
        corr_k = (labels == k).astype(np.float64)

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (conf_k >= lo) & (conf_k <= hi) if b == 0 else (conf_k > lo) & (conf_k <= hi)
            if mask.any():
                acc_b = corr_k[mask].mean()
                conf_b = conf_k[mask].mean()
                sce_terms.append(np.abs(acc_b - conf_b))
    sce = float(np.mean(sce_terms)) if sce_terms else 0.0

    rng = np.random.default_rng(0)
    ace_terms = []
    for r in range(n_resamples_ace):
        for k in range(K):
            conf_k = P[:, k]
            corr_k = (labels == k).astype(np.float64)
            if conf_k.size == 0:
                continue

            quantiles = np.linspace(0, 1, n_bins + 1)
            jitter = rng.uniform(-0.01, 0.01, size=quantiles.shape)
            q_edges_r = np.clip(np.quantile(conf_k, np.clip(quantiles + jitter, 0, 1)), 0, 1)
            q_edges_r = np.unique(q_edges_r)

            for b in range(len(q_edges_r) - 1):
                lo, hi = q_edges_r[b], q_edges_r[b + 1]
                mask = (conf_k >= lo) & (conf_k <= hi) if b == 0 else (conf_k > lo) & (conf_k <= hi)
                if mask.any():
                    acc_b = corr_k[mask].mean()
                    conf_b = conf_k[mask].mean()
                    ace_terms.append(np.abs(acc_b - conf_b))

    ace = float(np.mean(ace_terms)) if ace_terms else 0.0

    return {
        "ECE": ece,
        "MCE_raw": mce_raw,
        "MCE_filtered": mce_filtered,
        "SCE": sce,
        "ACE": ace,
        "bin_stats_fixed": stats_fixed,
        "bin_stats_adaptive": stats_adaptive,
    }


def plot_reliability_with_histogram(
    stats_fixed: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    stats_adaptive: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    save_path: str,
):
    _ensure_dir(os.path.dirname(save_path))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 1.5, 3]})

    # colors = ["darkorchid", "magenta", "dodgerblue"]
    colors = ["magenta", "dodgerblue"]


    # --- Left: Fixed bins (ECE/MCE) ---
    ax1.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    for (head, (bin_edges, bin_acc, bin_conf, bin_cnt)), color in zip(stats_fixed.items(), colors):
        mask = bin_cnt > 0
        ax1.plot(bin_conf[mask], bin_acc[mask], marker="o", linewidth=1.8, color=color, label=head.capitalize())
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence (per bin mean)")
    ax1.set_ylabel("Accuracy (per bin)")
    ax1.set_title("Fixed bins (ECE/MCE)")
    ax1.legend()

    # --- Middle: Histogram (sample distribution across bins) ---
    first_head = list(stats_fixed.keys())[0]
    edges_ref, _, _, bin_cnt_ref = stats_fixed[first_head]
    centers = 0.5 * (edges_ref[:-1] + edges_ref[1:])
    bin_width = (edges_ref[1] - edges_ref[0]) * 0.9

    # sum counts from all heads
    bin_cnt_total = np.zeros_like(bin_cnt_ref, dtype=float)
    for h in stats_fixed:
        _, _, _, bin_cnt = stats_fixed[h]
        bin_cnt_total += bin_cnt

    ax2.bar(
        centers,
        bin_cnt_total,
        width=bin_width,
        color="orchid",      # single color
        alpha=0.8,
        align="center",
    )

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence bin (fixed)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution")

    # --- Right: Adaptive bins (ACE) ---
    ax3.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    for (head, (bin_edges, bin_acc, bin_conf, bin_cnt)), color in zip(stats_adaptive.items(), colors):
        mask = bin_cnt > 0
        ax3.plot(bin_conf[mask], bin_acc[mask], marker="o", linewidth=1.8, color=color, label=head.capitalize())
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xlabel("Confidence (per bin mean)")
    ax3.set_ylabel("Accuracy (per bin)")
    ax3.set_title("Adaptive bins (ACE)")
    ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def effective_scale(cfg: DictConfig) -> float:
    return float(cfg.get("scale", 1.0))


def logits_to_conf_corr(logits: np.ndarray, labels: np.ndarray, scale: float):
    L = logits * scale
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L);  P /= P.sum(axis=1, keepdims=True)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    corr = (pred == labels).astype(np.float64)
    return conf, corr


@torch.no_grad()
def collect_logits(cfg: DictConfig, model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizer, split: str):
    device = next(model.parameters()).device
    dataset = get_dataset(cfg, tokenizer, split)

    if hasattr(dataset, "all_labels") and dataset.all_labels:
        label_vocab = list(dataset.all_labels)
    else:
        labels_list = getattr(dataset, "labels", [])
        label_vocab = sorted(set(labels_list))
    emotion2idx = {lbl: i for i, lbl in enumerate(label_vocab)}

    prompts = [get_prompt(lbl, cfg) for lbl in label_vocab]
    tok = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=cfg.dataset.max_text_length).to(device)
    C = model.class_text_encoder(tok)
    C = F.normalize(C, dim=-1)

    m = str(cfg.train.modality)
    use_audio = m in ["trimodal", "audio_text", "audio_only", "audio_text_unaligned"]
    use_text  = m in ["trimodal", "audio_text", "text_only", "audio_text_unaligned"]
    combine   = m in ["trimodal", "audio_text"]
    use_both  = (use_audio and use_text and combine)

    get_waveform  = lambda x: x[0]
    get_label     = lambda x: x[1]
    get_transcript= lambda x: x[2]

    out = {}
    if use_audio: out["audio"] = {"logits": []}
    if use_text:  out["text"]  = {"logits": []}
    if use_both:  out["both"]  = {"logits": []}
    labels_arr = []

    model.eval()
    for i in range(len(dataset)):
        s = dataset[i]
        y = emotion2idx[get_label(s)]
        labels_arr.append(int(y))

        z_a = None; z_t = None
        if use_audio:
            wav = get_waveform(s).unsqueeze(0).to(device)
            z_a = model.audio_encoder(wav)
        if use_text:
            sent = get_transcript(s)
            t_in = tokenizer(sent, return_tensors="pt", padding=True,
                             truncation=True, max_length=cfg.dataset.max_text_length)
            t_in = {k: v.to(device) for k, v in t_in.items()}
            z_t = model.input_text_encoder(t_in)

        if use_audio:
            logits_a = (z_a @ C.T).squeeze(0).cpu().numpy()
            out["audio"]["logits"].append(logits_a)
        if use_text:
            logits_t = (z_t @ C.T).squeeze(0).cpu().numpy()
            out["text"]["logits"].append(logits_t)
        if use_both:
            z_avg = F.normalize((z_a + z_t) / 2, dim=-1)
            logits_b = (z_avg @ C.T).squeeze(0).cpu().numpy()
            out["both"]["logits"].append(logits_b)

    out["labels"] = np.array(labels_arr, dtype=np.int64)
    return out

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    n_bins = int(cfg.get("n_bins", 10))
    eff_scale = effective_scale(cfg)
    split = cfg.get("split", "test")

    root = get_original_cwd()
    base_dir = os.path.join(root, "uncertainity")
    img_dir  = os.path.join(base_dir, "png")
    _ensure_dir(img_dir)

    model, tokenizer, device = load_trained_model(cfg)
    ds_name = cfg.dataset.model_checkpoint.split('/')[1][:-3]

    print(f"[Scale] effective scale: {eff_scale:.4f}")
    print(f"[Bins] n_bins: {n_bins}")

    data = collect_logits(cfg, model, tokenizer, split=split)
    labels = data["labels"]
    all_heads = [h for h in ["both", "audio", "text"] if h in data]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stats_fixed = {}
    stats_adaptive = {}

    print("\nCalibration Metrics")
    print(f"{'Head':<8} {'ECE':>8} {'MCE_raw':>10} {'MCE_filt':>10} {'SCE':>8} {'ACE':>8}")
    print("-" * 60)

    for head in all_heads:
        logits = np.stack(data[head]["logits"], axis=0)

        metrics = compute_calibration_metrics(
            logits, labels,
            n_bins=n_bins,
            min_bin_count_mce=5,
            n_resamples_ace=20,
            scale=eff_scale
        )

        ece   = metrics["ECE"] * 100
        mce_r = metrics["MCE_raw"] * 100
        mce_f = metrics["MCE_filtered"] * 100
        sce   = metrics["SCE"] * 100
        ace   = metrics["ACE"] * 100

        print(f"{head:<8} {ece:8.2f}% {mce_r:10.2f}% {mce_f:10.2f}% {sce:8.2f}% {ace:8.2f}%")

        stats_fixed[head]    = metrics["bin_stats_fixed"]
        stats_adaptive[head] = metrics["bin_stats_adaptive"]

    # combined plot (fixed vs adaptive)
    multi_title = f"Uncertainty (Fixed vs Adaptive bins + Distribution)"
    multi_fname = f"{ds_name}_uncertainty_fixed_adaptive_hist_{split}"
    multi_save = os.path.join(img_dir, f"{timestamp}_{multi_fname}.png")
    plot_reliability_with_histogram(stats_fixed, stats_adaptive, multi_title, multi_save)

    print(f"Combined uncertainty plot saved to: {multi_save}")



if __name__ == "__main__":
    main()
