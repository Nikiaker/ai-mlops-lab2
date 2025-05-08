import base64

# Read the binary file
with open('../checkpoints/best_model-epoch=15-val_loss=0.09.ckpt', 'rb') as file:
    binary_data = file.read()

# Encode binary data to Base64 string
encoded_string = base64.b64encode(binary_data).decode('utf-8')

with open('../checkpoints/encoded_model.py', 'w') as file:
    file.write(f"checkpoint_content = \"{encoded_string}\"")