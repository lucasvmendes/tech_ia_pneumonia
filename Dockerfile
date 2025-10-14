# Usamos a imagem oficial do TensorFlow (CPU) para evitar problemas de compatibilidade
FROM tensorflow/tensorflow:2.17.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copia apenas requirements e src
COPY src/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# criar pastas de runtime (serão normalmente montadas via volume)
RUN mkdir -p /app/data /app/models /app/reports

# Default command: treinar (pode ser sobrescrito ao rodar o container)
CMD ["python", "src/main.py"]
