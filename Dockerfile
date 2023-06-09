# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file to the working directory
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the remaining code to the working directory
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Run the FastAPI application using Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
