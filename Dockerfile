# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 for the API
EXPOSE 8000

# Define environment variable
ENV MODEL_PATH=models/entity_resolution_model.pkl
ENV DB_PATH=data/clients.db

# Run app.py when the container launches
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
