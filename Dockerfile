FROM python:3.11-slim

# Create non-root user
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Install deps first (leverages Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and artifacts
COPY . .

# Switch to non-root
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
