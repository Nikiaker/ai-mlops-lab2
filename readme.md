bentoml serve bento:svc
docker build -t my-bentoml-lambda:latest .
docker run -p 9000:8080 my-bentoml-lambda:latest

aws s3 cp img_creator/img.jpg s3://my-ml-model-bucket3/inputs/img.jpg