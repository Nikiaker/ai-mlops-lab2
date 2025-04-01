import bentoml
from bentoml.io import JSON
import torch
from main import LightningModel


# Load your model from the PyTorch Lightning checkpoint
CHECKPOINT_PATH = "./checkpoints/best_model-epoch=03-val_loss=0.15.ckpt"
model = LightningModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

# Create a BentoML service
svc = bentoml.Service("model_service")

# Define an API endpoint
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    """
    Expects input_data to be a JSON object with the necessary fields for the model.
    """
    # Preprocess input data
    input_tensor = torch.tensor(input_data["data"])
    
    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Postprocess and return the result
    return {"prediction": prediction.tolist()}