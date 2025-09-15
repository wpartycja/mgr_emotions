import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from omegaconf import OmegaConf
from pathlib import Path

import os
from dotenv import load_dotenv
from datascripts.dataset_loader import get_dataset_and_collate_fn
from model.clap_trimodal import CLAPTriModal
from checkpoint import load_checkpoint
from model_loader import load_trained_model
from omegaconf import DictConfig
import hydra
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pathlib import Path
import numpy as np

from umap import UMAP
import plotly.graph_objects as go
from hydra.utils import to_absolute_path
import plotly.graph_objects as go
from hydra.utils import to_absolute_path
from pathlib import Path
from hydra.utils import to_absolute_path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


def extract_embeddings(cfg, modality, split="val", max_batches=20):
    """
    Extract embeddings for a given modality ("audio", "text", "both").
    Returns: embeddings [N, D], labels [N]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)
    dataset, collate_fn = get_dataset_and_collate_fn(cfg, tokenizer, split)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                        shuffle=False, collate_fn=collate_fn)

    model = CLAPTriModal(
        cfg.model.audio_encoder, cfg.model.text_encoder,
        d_proj=cfg.model.d_proj, access_token=access_token,
        init_tau=cfg.model.init_tau,
        min_logit_scale=cfg.model.min_logit_scale,
        max_logit_scale=cfg.model.max_logit_scale,
        dropout_rate=0
    ).to(device)
    model.eval()

    if cfg.dataset.get("model_checkpoint"):
        model, _, _= load_trained_model(cfg)

    embeddings, labels = [], []

    with torch.no_grad():
        for i, (audio, text_inputs, class_text_inputs, _) in enumerate(loader):
            if i >= max_batches:  # limit for speed
                break

            audio = audio.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            out = model(audio=audio, input_text=text_inputs)

            if modality == "audio":
                emb = out["audio_embed"]
            elif modality == "text":
                emb = out["input_text_embed"]
            elif modality == "both":
                emb = torch.cat([out["audio_embed"], out["input_text_embed"]], dim=-1)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            embeddings.append(emb.cpu())
            labels.extend(dataset.labels[i * cfg.train.batch_size:(i + 1) * cfg.train.batch_size])

    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings, labels


def plot_latent_space(embeddings, labels, method, modality, dataset,
                      outdir="png", title=None, view="default"):
    """
    Runs dimensionality reduction and saves figure.
    - PCA, t-SNE, UMAP, LDA -> Matplotlib 2D PNG
    - UMAP3D -> Plotly 3D HTML
    """

    color_map = {
        "angry": "red",
        "happy": "lightcoral",  
        "neutral": "lightblue",
        "sad": "darkblue",
    }

    method_name = None
    X_embedded = None

    if method == "tsne":
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        method_name = "t-SNE"
    elif method == "pca":
        X_embedded = PCA(n_components=2).fit_transform(embeddings)
        method_name = "PCA"
    elif method == "umap":
        X_embedded = UMAP(n_components=2, init="random", random_state=42).fit_transform(embeddings)
        method_name = "UMAP"
    elif method == "lda":
        X_embedded = LDA(n_components=2).fit_transform(embeddings, labels)
        method_name = "LDA"
    elif method == "umap3d":
        umap_3d = UMAP(n_components=3, init="random", random_state=42)
        X_embedded = umap_3d.fit_transform(embeddings)
        plot_3d_space(X_embedded, labels, modality, dataset, view, color_map)
        return
    else:
        raise ValueError("Wrong method name, not available")


    Path(outdir).mkdir(exist_ok=True, parents=True)
    save_path = Path(outdir) / f"{dataset.lower()}/{modality}_{method}.png"

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    label=lab, alpha=0.6, color=color_map.get(lab, "gray"))
    plt.legend()
    plt.title(title or f"{method_name} of {modality} embeddings in {dataset.rstrip('4Cls')}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_3d_space(X_embedded, labels, modality, dataset, view, color_map, title=None):
    traces = []
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        traces.append(
            go.Scatter3d(
                x=X_embedded[idx, 0], y=X_embedded[idx, 1], z=X_embedded[idx, 2],
                mode="markers", name=lab,
                marker=dict(size=3, opacity=0.7, color=color_map.get(lab, "gray")),
            )
        )

    cameras = {
        "default": dict(eye=dict(x=1.75, y=1.75, z=1.25)),
        "23": dict(eye=dict(x=0.01, y=2.2, z=2.2)),
        "13": dict(eye=dict(x=2.2, y=0.01, z=2.2)),
        "12": dict(eye=dict(x=2.2, y=2.2, z=0.01)),
        "31": dict(eye=dict(x=2.2, y=0.01, z=2.2)),
    }
    cam = cameras.get(view, cameras["default"])

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or f"UMAP 3D of {modality} embeddings in {dataset}",
        scene=dict(
            xaxis_title="x-axis",
            yaxis_title="y-axis",
            zaxis_title="z-axis",
        ),
        legend=dict(itemsizing="constant"),
        scene_camera=cam,
    )

    out_dir_abs = Path(to_absolute_path(f"png/{dataset.lower()}/html"))
    out_dir_abs.mkdir(parents=True, exist_ok=True)
    html_path = out_dir_abs / f"{modality}_umap3d.html"
    fig.write_html(str(html_path))
    print(f"Saved HTML:\n  {html_path}")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_space_visualisation(cfg: DictConfig):
    
    for modality in ['text', 'audio', 'both']:
        embeddings, labels = extract_embeddings(cfg, modality=modality, split="test", max_batches=10)
        for method in ['pca', 'tsne', 'lda', 'umap', 'umap3d']:
            plot_latent_space(embeddings, labels, method=method, modality=modality, dataset=cfg.dataset.name, outdir=f"png/")

if __name__ == "__main__":
    run_space_visualisation()
