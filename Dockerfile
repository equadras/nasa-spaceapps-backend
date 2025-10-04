# Use uma imagem base do Python
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala as dependências
COPY requirements.txt .
# --no-cache-dir acelera e deixa a imagem menor
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código do backend
COPY . .

# Comando padrão (será sobrescrito para a API, mas é bom ter)
CMD ["echo", "Backend service ready. Run 'loader.py' or start the API."]
