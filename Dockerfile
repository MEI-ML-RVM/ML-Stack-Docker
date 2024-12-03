# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive mode for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-opencv ffmpeg libsm6 libxext6 libxrender-dev tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy files
COPY app /app

# Copy requirements file for dependency installation
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 8765

# Command to run the script
CMD ["python3", "script.py"]
