import json, boto3
from bento import InferenceService
from PIL import Image
from torchvision import transforms
import json

s3 = boto3.client("s3")

def preprocess_image(image_path: str) -> list:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Convert tensor to a list
    tensor_list = tensor_image.squeeze(0).numpy().tolist()
    return tensor_list

def lambda_handler(event, context):
    print("Event: %s", event)
    print("Context: %s", context)

    # 1. Parse bucket/key
    record = event["Records"][0]["s3"]
    bucket = record["bucket"]["name"]
    key = record["object"]["key"]
    # 2. Download image as bytes and preprocess
    tmp_path = f"/tmp/{key.split('/')[-1]}"
    s3.download_file(bucket, key, tmp_path)
    data = preprocess_image(tmp_path)
    # 3. Call BentoML service
    service = InferenceService()
    response = service.predict(data)
    # 4. Save results back to S3
    result_key = key.replace("inputs/", "outputs/") + ".json"
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(response),
        ContentType="application/json"
    )
    return {"statusCode": 200, "body": json.dumps(response)}