FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Evita escribir .pyc y activa logging directo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip libgl1 libglib2.0-0 \
    && apt-get clean

# Crea el directorio de trabajo
WORKDIR /app

RUN mkdir -p /app/datasets

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "A-J", "train"]

# docker build -t train-nets .
# docker run -it --rm --gpus all train-nets