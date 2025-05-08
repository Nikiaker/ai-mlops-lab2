FROM public.ecr.aws/lambda/python:3.9

# Install dependencies
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

# Copy model weights and service code
COPY checkpoints/encoded_model.py ./checkpoints/
COPY service.py bento.py ./
COPY lib/data_model.py ./lib/

# Set the CMD to your handler
CMD ["service.lambda_handler"]
