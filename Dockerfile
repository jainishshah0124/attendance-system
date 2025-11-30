FROM jainish0124/attendance-system:v3
WORKDIR /app
COPY . .
#ENV FLASK_APP=app.py
RUN pip install -r requirements.txt
#EXPOSE 8080
#CMD ["python", "app.py"]

# Cloud Run expects the server to listen on port 8080
ENV PORT 8080

# Ensure logs are sent to stdout/stderr (needed for Cloud Run logging)
ENV PYTHONUNBUFFERED=1

# Expose Cloud Run port
EXPOSE 8080

# Start Gunicorn with logging enabled
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--log-level", "debug", "--access-logfile", "-", "app:app"]