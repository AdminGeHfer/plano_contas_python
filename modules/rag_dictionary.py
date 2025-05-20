import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/dicionario_embeddings.csv"
INDEX_PATH = "data/dicionario_faiss.index"

# Carregamento único
df_dicionario = pd.read_csv(CSV_PATH, delimiter=";")
index_faiss = faiss.read_index(INDEX_PATH)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Pré-carregamento dos vetores de todo o corpus
all_embeddings = np.array([index_faiss.reconstruct(i) for i in range(index_faiss.ntotal)])

def consultar_exemplos_similares(texto_usuario, conta_sugerida, k=5, busca_global_k=50):
  query_vec = model.encode([texto_usuario])

  # 1º – busca no corpus inteiro
  D, I = index_faiss.search(np.array(query_vec), busca_global_k)
  candidatos = df_dicionario.iloc[I[0]].copy()
  candidatos["dist"] = D[0]

  # 2º – se a conta sugerida aparecer no top-k global, retorna os exemplos dela;
  #      caso contrário devolve os melhores k globais para que o LLM decida.
  subset = candidatos[candidatos["Conta Gerencial"] == conta_sugerida]
  if not subset.empty:
    return subset.nlargest(k, "dist")["Exemplo de Uso"].tolist()

  return candidatos.nlargest(k, "dist")["Exemplo de Uso"].tolist()
