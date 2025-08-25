import os
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from datetime import datetime


def get_optuna_filename(dataset_name: str, run_timestamp: str, suffix: str = "results.json") -> str:
    orig_dir = HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else os.getcwd()
    base_dir = os.path.abspath(os.path.join(orig_dir, "..", "..", "..", "optuna_outputs"))
    dataset_dir = os.path.join(base_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(dataset_dir, f"{run_timestamp}_{suffix}")


def init_run_log_file(cfg: DictConfig, file_path: str, run_timestamp: str):
    """Create the run file if it doesn't exist, with a single config header."""
    if os.path.exists(file_path):
        return  # keep existing header if you’re appending more trials in the same run/file

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    payload = {
        "run": {
            "timestamp": run_timestamp,
            "dataset": str(cfg.dataset.name) if "dataset" in cfg and "name" in cfg.dataset else None,
            "config": config_dict,
        },
        "trials": [],
    }
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)


def start_trial_entry(file_path: str, trial_number: int, params: dict):
    with open(file_path, "r") as f:
        data = json.load(f)

    # if trial already exists (e.g., resumed), don't duplicate
    for t in data.get("trials", []):
        if t.get("trial") == trial_number:
            # update params if needed
            t["params"] = params
            break
    else:
        data.setdefault("trials", []).append({
            "trial": trial_number,
            "params": params,
            "results": []
        })

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def log_epoch_results(file_path: str, trial_number: int, epoch: int,
                      acc_both: float, acc_audio: float, acc_text: float):
    with open(file_path, "r") as f:
        data = json.load(f)

    for t in data.get("trials", []):
        if t.get("trial") == trial_number:
            t["results"].append({
                "epoch": epoch,
                "accuracy_both": acc_both,
                "accuracy_audio": acc_audio,
                "accuracy_text": acc_text,
            })
            break
    else:
        raise ValueError(f"Trial {trial_number} not found in {file_path}")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def log_best_trial(file_path: str, best_trial):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    per_modality = {
        "audio+text": best_trial.user_attrs.get("acc_audio_text"),
        "audio":   best_trial.user_attrs.get("acc_audio"),
        "text":    best_trial.user_attrs.get("acc_text"),
    }

    data["best_trial"] = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "per_modality": per_modality
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def log_best_per_modality(file_path: str, trials):
    """
    Finds the best trial separately for each modality and writes:
    """

    # Helper to pick best trial for a given user_attr key
    def pick_best(key):
      candidates = [trial for trial in trials if key in trial.user_attrs]
      if not candidates:
          return None
      best = max(candidates, key=lambda trial: trial.user_attrs[key])
      return {
          "trial": best.number,
          "value": best.user_attrs[key],
          "params": best.params
      }

    bests = {
        "audio+text": pick_best("acc_audio_text"),
        "audio":   pick_best("acc_audio"),
        "text":    pick_best("acc_text"),
    }

    with open(file_path, "r") as f:
        data = json.load(f)
    data["best_by_modality"] = bests
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
