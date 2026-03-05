# Dockerfile

# 1) Base image: Python 3.11 on a slim Debian variant
FROM python:3.11-slim

# 2) Set working directory inside the container
WORKDIR /app

# 3) Some Python env tweaks
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4) (Optional) Install system deps if you need build tools or curl
# Uncomment if something fails due to missing build tools:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# 5) Copy only requirements first — good for Docker layer caching
COPY requirements.txt .

# 6) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7) Now copy the rest of your project (code, models, etc.)
COPY . .

# 8) Expose the port FastAPI/uvicorn will use
EXPOSE 8000

# 9) Default command: run your FastAPI app via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
