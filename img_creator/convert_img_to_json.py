from PIL import Image
from torchvision import transforms
import json
import numpy as np

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open("img.png").convert("L")
tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

# Convert tensor to a list
tensor_list = tensor_image.squeeze(0).numpy().tolist()
dic = {"data": tensor_list}

# Save to JSON file
with open("tensor_image.json", "w") as json_file:
    json.dump(dic, json_file)

print(tensor_image)