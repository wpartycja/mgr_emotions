import optuna
import hydra
import os
from omegaconf import OmegaConf
from train import train  # import your train function

@hydra.main(config_path="conf", config_name="config", version_base=None)
def optuna_runner(cfg):
    def objective(trial):
        os.environ["WANDB_MODE"] = "disabled"
        
        cfg.train.lr_proj = trial.suggest_float("lr_proj", 1e-5, 1e-2, log=True)
        cfg.train.lr_enc = trial.suggest_float("lr_enc", 1e-5, 1e-3, log=True)
        cfg.model.d_proj = trial.suggest_categorical("d_proj", [128, 256, 512])
        cfg.train.max_sharpness = trial.suggest_float("sharpness", 0.1, 5.0)


        return train(cfg, return_val_metric=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    print("Best trial:", study.best_trial)

if __name__ == "__main__":
    optuna_runner()
