import os
import torch
import hydra

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
from evaluation import evaluate
from model.contrastive_loss import clip_contrastive_loss
from model_loader import load_class_embeds
from datascripts.dataset_loader import get_dataset, get_dataset_and_collate_fn
from model.clap_trimodal import CLAPTriModal

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()
access_token = os.getenv("HF_TOKEN")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    print("Loading tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=access_token)

    print("Loading dataset...")
    # Load dataset and collate function
    train_dataset, collate_fn = get_dataset_and_collate_fn(cfg, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = get_dataset(cfg, tokenizer, "validation")
    label_names = val_dataset.all_labels

    print(f"Train set size: {len(train_dataset)} samples")
    print(f"Validation set size: {len(val_dataset)} samples")

    # Initialize model
    print("Initializing model...")
    model = CLAPTriModal(
        cfg.model.audio_encoder, cfg.model.text_encoder, d_proj=cfg.model.d_proj, access_token=access_token
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    print("Starting training...")
    for epoch in range(cfg.train.epochs):
        print(f"\n------ Epoch {epoch} ------\n")

        model.train()
        total_loss, total_audio_text, total_audio_class, total_text_class = 0, 0, 0, 0

        for step, (audio, text_inputs, class_text_inputs) in enumerate(train_loader):
            audio = audio.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            class_text_inputs = {k: v.to(device) for k, v in class_text_inputs.items()}

            out = model(audio=audio, input_text=text_inputs, class_text=class_text_inputs)

            scale = out["contrastive_scale"]
            audio_embed = out["audio_embed"]
            text_embed = out["input_text_embed"]
            class_embed = out["class_text_embed"]

            # Contrastive losses between all 3 modalities (pairwise)
            loss_audio_text = clip_contrastive_loss(audio_embed, text_embed, scale, device=device)
            loss_audio_class = clip_contrastive_loss(audio_embed, class_embed, scale, device=device)
            loss_text_class = clip_contrastive_loss(text_embed, class_embed, scale, device=device)

            # Proportional loss weights
            losses = [loss_audio_text, loss_audio_class, loss_text_class]
            temperature = 0.5
            loss_tensor = torch.tensor([l.item() for l in losses])
            weights = torch.softmax(loss_tensor / temperature, dim=0)
            loss = sum(w * l for w, l in zip(weights, losses))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_audio_text += loss_audio_text.item()
            total_audio_class += loss_audio_class.item()
            total_text_class += loss_text_class.item()

            if step % 10 == 0:
                print(f"Step {step:03d}")
                print(
                    f"Loss: {loss.item():.4f} | A↔T: {loss_audio_text.item():.4f} | A↔C: {loss_audio_class.item():.4f} | T↔C: {loss_text_class.item():.4f}"
                )
                weight_audio_text, weight_aduio, weight_text = weights
                print(
                    f"Loss weights | A↔T: {weight_audio_text:.4f} | A↔C: {weight_aduio:.4f} | T↔C: {weight_text:.4f}"
                )

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_audio_text_loss = total_audio_text / len(train_loader)
        avg_audio_loss = total_audio_class / len(train_loader)
        avg_text_loss = total_text_class / len(train_loader)
        print(f"\nAvg Loss: {avg_loss:.4f}")
        print(f"Audio + Text: {avg_audio_text_loss:.4f} | Audio: {avg_audio_loss:.4f} | Text: {avg_text_loss:.4f}")

        # Evaluation after each epoch
        class_embeds, emotion2idx, idx2emotion = load_class_embeds(cfg, model, tokenizer, label_names, device)
        evaluate(cfg, model, tokenizer, val_dataset, class_embeds, emotion2idx, idx2emotion, device)

    # Save model
    path = f"./weights/{cfg.datasets.model_output}"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    train()
