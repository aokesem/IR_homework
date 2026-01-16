# Use a pipeline as a high-level helper
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")