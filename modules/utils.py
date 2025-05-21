import re
from sentence_transformers import SentenceTransformer

modelo_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def gerar_embeddings(textos):
  if isinstance(textos, str):
    textos = [textos]
  return modelo_st.encode(textos, show_progress_bar=False).tolist()

def normalizar_texto(textos):
  if isinstance(textos, str):
    textos = [textos]
  return [re.sub(r"[^a-z0-9\s]", "", t.lower()) for t in textos]
