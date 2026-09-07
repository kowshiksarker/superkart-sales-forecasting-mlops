FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from creating .pyc files
# and ensure logs are displayed immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy deployment dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit application
COPY app.py .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
