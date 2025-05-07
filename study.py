import optuna
import json
import uuid
import pytorch_lightning as pl
import torch

from lib.data_model import Data, LightningModel

N_TRIALS = 1
MAX_EPOCHS = 5

# Hyperparameter Optimization with Optuna
def objective(trial: optuna.Trial) -> float:
    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    #learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    learning_rate = trial.suggest_float("learning_rate", 0, 1)

    data_module = Data(batch_size=64)
    model = LightningModel(learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=False, enable_checkpointing=False)
    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print("Setting float32 matmul precision to medium")
        torch.set_float32_matmul_precision("medium")

    print("##### STUDYING BEST HYPERPARAMETERS #####")
    # Hyperparameter Optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    print("##### STUDYING BEST HYPERPARAMETERS FINISHED #####")
    print(f"Best hyperparameters: {best_params}")

    # Save the best hyperparameters to a file
    filename = f"hyperparameters/best_hyperparameters_{uuid.uuid4().hex}.json"
    with open(filename, "w") as f:
        json.dump(best_params, f, indent=4)