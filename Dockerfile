FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY train.py app.py ./
COPY templates ./templates
COPY static ./static

# Train the model while building the image, so the image ships with model.joblib
RUN python train.py

EXPOSE 8000
# Render and other hosts pass the port in $PORT; 8000 is the local default
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} 'app:create_app()'"]
