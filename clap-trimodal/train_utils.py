import torch

def freeze_for_modality(model, modality: str):
    """
    Freeze only the encoders that won't be used in this training mode.
    - trimodal:      all trainable
    - audio_text:    freeze class_text encoder
    - audio_only:    freeze input_text encoder (train audio + class)
    - text_only:     freeze audio encoder (train text + class)
    """
    # unfreeze everything first
    for enc in [model.audio_encoder, model.input_text_encoder, model.class_text_encoder]:
        for p in enc.parameters():
            p.requires_grad = True

    if modality == "trimodal":
        model.audio_encoder.train()
        model.input_text_encoder.train()
        model.class_text_encoder.train()

    elif modality == "audio_text":
        model.audio_encoder.train()
        model.input_text_encoder.train()
        for p in model.class_text_encoder.parameters():
            p.requires_grad = False
        model.class_text_encoder.eval()

    elif modality == "audio_only":
        model.audio_encoder.train()
        model.class_text_encoder.train()
        for p in model.input_text_encoder.parameters():
            p.requires_grad = False
        model.input_text_encoder.eval()

    elif modality == "text_only":
        model.input_text_encoder.train()
        model.class_text_encoder.train()
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        model.audio_encoder.eval()
    
    elif modality == "audio_text_unaligned":
        model.input_text_encoder.train()
        model.audio_encoder.train()
        model.class_text_encoder.train()

    else:
        raise ValueError(f"Unknown modality: {modality}")


def build_optimizer_from_trainable(model, cfg):
    """
    Build param groups strictly from params with requires_grad=True.
    Ensures frozen encoders are excluded from the optimizer.
    """
    def trainable(ps):
        return [p for p in ps if p.requires_grad]

    param_groups = [
        {"params": trainable(model.audio_encoder.parameters()),       "lr": cfg.train.lr_enc},
        {"params": trainable(model.input_text_encoder.parameters()),  "lr": cfg.train.lr_enc},
        {"params": trainable(model.class_text_encoder.parameters()),  "lr": cfg.train.lr_enc},
        {"params": [p for p in [model.logit_scale] if p.requires_grad], "lr": cfg.train.lr_proj},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.train.lr_proj)
