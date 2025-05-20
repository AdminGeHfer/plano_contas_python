from sentence_transformers import SentenceTransformer

def get_embedding_model():
  return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
