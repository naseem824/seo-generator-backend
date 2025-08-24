# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

# Copy the rest of the app code
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5001

# Use environment variable PORT if available (Fly.io sets this automatically)
ENV PORT=5001

# Start the Flask app
CMD ["python", "app.py"]
