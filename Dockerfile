# Use a lightweight Python image as the base
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port your Flask app will run on (5000)
EXPOSE 5000

# Set the command to run your Flask app
CMD ["python", "app.py"]
