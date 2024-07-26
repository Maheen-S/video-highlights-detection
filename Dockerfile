# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . .

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Make port 8501 available to the world outside this container
# EXPOSE 8501

# # Run app.py when the container launches
# # CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT  ["streamlit", "run", "video_highlight_detection/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies for OpenCV and other necessary libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install the spaCy model
RUN python -m spacy download en_core_web_sm

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "video_highlight_detection/app.py", "--server.port=8501", "--server.address=0.0.0.0"]




# Build the Docker Image:
# docker build -t cricket-highlight-detector .

# Run the Docker Container:
# docker run -p [host_port]:[container_port] [image_name]
# docker run -p 8502:8501 cricket-highlight-detector