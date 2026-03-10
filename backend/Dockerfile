FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models during build (cached in Docker layer)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Verify bm25_index.pkl exists
RUN python -c "import os; assert os.path.exists('bm25_index.pkl'), 'bm25_index.pkl not found'"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
