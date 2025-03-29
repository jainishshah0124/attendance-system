FROM jainish0124/attendance-system:v3
WORKDIR /app
COPY . .
ENV FLASK_APP=app.py
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]