# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This assumes the Docker build context is the 'docuai' directory.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces image size.
# Consider creating a non-root user for security best practices in production.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
# This copies the content of 'docuai/app' into '/app/app' in the container.
COPY ./app /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable to help Python find modules in the /app directory.
# The application structure is /app/app, so modules are like app.main, app.core etc.
ENV PYTHONPATH=/app

# Run uvicorn when the container launches
# The --reload flag is useful for development but typically removed for production.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
