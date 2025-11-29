# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- Dependency Installation ---
# Copy the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy the application code into the container.
# This assumes the 'lib' directory is located alongside 'main.py'.
COPY . .

# --- Execution ---
# Run the worker script.
# The -u flag ensures that logs are sent straight to stdout without buffering,
# which is essential for real-time log monitoring in container orchestrators.
CMD [ "python", "-u", "main.py" ]
