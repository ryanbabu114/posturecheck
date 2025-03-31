# Use an official Python runtime as base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your local folder to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run Flask using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 posture_correction:app
