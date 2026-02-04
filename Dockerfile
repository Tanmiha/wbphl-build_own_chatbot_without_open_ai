# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
