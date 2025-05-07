import bentoml
from bentoml.io import JSON
import torch
from train import LightningModel

CHECKPOINT_PATH = "./checkpoints/best_model-epoch=03-val_loss=0.15.ckpt"
model = LightningModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

svc = bentoml.Service("model_service")

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    input_tensor = torch.tensor(input_data["data"])
    
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_digit = predictions.argmax(dim=1).item()
    
    return {
        "predicted_digit": predicted_digit,
        "prediction_values": predictions.tolist(),
    }