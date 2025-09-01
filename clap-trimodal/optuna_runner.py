import optuna
import hydra
import os
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from train import train
from utils.optuna_log import (
    get_optuna_filename, init_run_log_file, start_trial_entry, log_best_trial, log_best_per_modality
)
from datetime import datetime


@hydra.main(config_path="conf", config_name="config", version_base=None)
def optuna_runner(cfg):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = get_optuna_filename(cfg.dataset.name, run_timestamp, suffix="results.json")
    init_run_log_file(cfg, log_file, run_timestamp)

    def objective(trial):
        os.environ["WANDB_MODE"] = "disabled"

        cfg.train.lr_proj = trial.suggest_float("lr_proj", 1e-5, 1e-2, log=True)
        cfg.train.lr_enc  = trial.suggest_float("lr_enc",  1e-6, 1e-3, log=True)
        # cfg.model.d_proj = trial.suggest_categorical("d_proj", [128, 256, 512])
        # cfg.train.max_sharpness = trial.suggest_float("sharpness", 0.1, 5.0)

        start_trial_entry(log_file, trial.number, {"lr_proj": cfg.train.lr_proj, "lr_enc": cfg.train.lr_enc})

        val_metric = train(cfg, return_val_metric=True, trial=trial, trial_number=trial.number, run_timestamp=run_timestamp)

        # store in trial for easy access from study.best_trial
        trial.set_user_attr("best_val_acc",  val_metric["best_val_acc"])
        trial.set_user_attr("acc_audio_text",  val_metric["acc_audio_text"])
        trial.set_user_attr("acc_text", val_metric["acc_text"])
        trial.set_user_attr("acc_audio",  val_metric["acc_audio"])

        return val_metric["best_val_acc"]

    sampler = TPESampler(multivariate=True)
    pruner = MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=20, n_jobs=1)

    print(f"\nOptuna run finished. Best trial: {study.best_trial.number}")
    print(f"Best accuracy mean (bimodal + text + audio)/3 : {study.best_trial.value}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    log_best_trial(log_file, study.best_trial)
    log_best_per_modality(log_file, study.trials)

if __name__ == "__main__":
    optuna_runner()
