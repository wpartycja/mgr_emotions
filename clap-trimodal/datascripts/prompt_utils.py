from omegaconf import DictConfig


def get_prompt(label: str, cfg: DictConfig, **kwargs) -> str:
    """
    Generate a prompt for a given label using either:
    - a per-class dictionary from cfg.dataset.label2text
    - or a string template from cfg.dataset.prompt_template
    """

    # Option 1: custom per-class dictionary
    if hasattr(cfg.dataset, "label2text") and cfg.dataset.label2text:
        label_prompts = cfg.dataset.label2text
        if label not in label_prompts:
            raise ValueError(f"No prompt defined for label '{label}' in label2text.")
        return label_prompts[label]

    # Option 2: use formatable template
    if hasattr(cfg.dataset, "prompt_template") and cfg.dataset.prompt_template:
        template = cfg.dataset.prompt_template
        return template.format(label=label, **kwargs)

    raise ValueError("No prompt source defined: set either `label2text` or `prompt_template` in config.")


# def get_prompt(emotion, cfg):
#     """Enhanced prompts for better emotion separation"""
#     enhanced_prompts = {
#         'happy': "This person sounds extremely joyful, cheerful, and excited with bright positive energy",
#         'sad': "This person sounds deeply sorrowful, melancholic, and depressed with heavy negative emotion", 
#         'angry': "This person sounds furious, hostile, and aggressive with intense burning rage",
#         'neutral': "This person sounds completely calm, flat, and emotionally neutral with no expression"
#     }
    
#     # Use enhanced prompt if available, otherwise fallback to original
#     return enhanced_prompts.get(emotion.lower(), f"This audio expresses {emotion}")
