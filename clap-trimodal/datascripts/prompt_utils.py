from omegaconf import DictConfig


def get_prompt(label: str, cfg: DictConfig, **kwargs) -> str:
    """
    Generate a prompt for a given label using either:
    - a per-class dictionary from cfg.datasets.label2text
    - or a string template from cfg.datasets.prompt_template
    """

    # Option 1: custom per-class dictionary
    if hasattr(cfg.datasets, "label2text") and cfg.datasets.label2text:
        label_prompts = cfg.datasets.label2text
        if label not in label_prompts:
            raise ValueError(f"No prompt defined for label '{label}' in label2text.")
        return label_prompts[label]

    # Option 2: use formatable template
    if hasattr(cfg.datasets, "prompt_template") and cfg.datasets.prompt_template:
        template = cfg.datasets.prompt_template
        return template.format(label=label, **kwargs)

    raise ValueError("No prompt source defined: set either `label2text` or `prompt_template` in config.")
