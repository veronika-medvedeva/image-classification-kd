FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]
