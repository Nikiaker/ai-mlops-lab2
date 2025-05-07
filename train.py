import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from lib.data_model import Data, LightningModel


def predict_image(model_checkpoint: str, image_path: str) -> int:
    model = LightningModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(tensor_image)
        prediction = logits.argmax(dim=1).item()

    print(f"Predicted digit: {prediction}")
    return prediction

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print("Setting float32 matmul precision to medium")
        torch.set_float32_matmul_precision("medium")


    # Hyperparameter tuning with Optuna
    best_params = {
        "learning_rate": 0.000751955129580044,
        "batch_size": 64
    }

    # Model Monitoring with WandbLogger
    wandb_logger = WandbLogger(project="ai-mlops-lab1")

    # Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="checkpoints",
        filename="best_model-{epoch:02d}-{val_loss:.2f}" 
    )

    # Train best model
    best_model = LightningModel(learning_rate=best_params["learning_rate"])
    best_data_module = Data(batch_size=best_params["batch_size"])
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    trainer.fit(best_model, best_data_module)
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_checkpoint_path}")

    # Test the best model
    trainer.test(best_model, dataloaders=best_data_module.test_dataloader())
