import optuna
import hydra
import os
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from omegaconf import OmegaConf
from train import train

@hydra.main(config_path="conf", config_name="config", version_base=None)
def optuna_runner(cfg):
    def objective(trial):
        os.environ["WANDB_MODE"] = "disabled"
        
        cfg.train.lr_proj = trial.suggest_float("lr_proj", 1e-5, 1e-3, log=True)
        cfg.train.lr_enc = trial.suggest_float("lr_enc", 1e-5, 1e-3, log=True)
        # cfg.model.d_proj = trial.suggest_categorical("d_proj", [128, 256, 512]) # zrób samemu
        # cfg.train.max_sharpness = trial.suggest_float("sharpness", 0.1, 5.0) # sprawdz jak idzie adaptowalnemu i zaleznie od tego optuna


        return train(cfg, return_val_metric=True, trial=trial)

    sampler = TPESampler(multivariate=True)
    pruner = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=100, n_jobs=1)  

    # Output best result
    print("Best trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    optuna_runner()
