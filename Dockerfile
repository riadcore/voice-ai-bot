# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python dependencies (including TTS)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 5000

# Run app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
