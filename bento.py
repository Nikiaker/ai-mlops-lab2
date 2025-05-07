import bentoml
import torch
from train import LightningModel

CHECKPOINT_PATH = "./checkpoints/best_model-epoch=15-val_loss=0.09.ckpt"
model = LightningModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

@bentoml.service
class InferenceService:
    @bentoml.api
    def predict(self, data: list) -> dict:
        input_tensor = torch.tensor(data)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            predictions = model(input_tensor)
            predicted_digit = predictions.argmax(dim=1).item()
        
        return {
            "predicted_digit": predicted_digit,
            "prediction_values": predictions.tolist(),
        }