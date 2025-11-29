# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- Dependency Installation ---
# Copy the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy the application code into the container
COPY api.py .

# --- Runtime Configuration ---
# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uvicorn.
# The host 0.0.0.0 is crucial for Docker to expose the port correctly.
CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000" ]
