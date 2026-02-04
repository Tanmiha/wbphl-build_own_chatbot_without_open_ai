FROM python:3.10

WORKDIR /app

COPY . .

# Upgrade pip FIRST (critical)
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "server.py"]
