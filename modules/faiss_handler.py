import faiss
import pandas as pd
import numpy as np
from modules.embeddings_model import get_embedding_model

INDEX_PATH = "data/faiss_index.index"
CSV_PATH = "data/dados_enriquecidos.csv"

def criar_index_faiss(csv_origem="data/dataset_plano_de_contas.csv"):
  model = get_embedding_model()
  df = pd.read_csv(csv_origem, delimiter=";", encoding="utf-8")
  df["texto_base"] = df["DESC_PLA"] + " - " + df["GRUPO"]
  df = df.dropna(subset=["texto_base"]).drop_duplicates(subset=["texto_base"])

  corpus = ["Conta gerencial para " + texto.lower() for texto in df["texto_base"]]
  embeddings = model.encode(corpus, show_progress_bar=True)

  index = faiss.IndexFlatL2(embeddings.shape[1])
  index.add(np.array(embeddings))

  # Salvar artefatos
  faiss.write_index(index, INDEX_PATH)
  df.to_csv(CSV_PATH, sep=";", index=False)
  print("Índice FAISS criado e salvo com sucesso.")

def carregar_index_faiss():
  model = get_embedding_model()
  index = faiss.read_index(INDEX_PATH)
  df = pd.read_csv(CSV_PATH, delimiter=";", encoding="utf-8")
  return index, df, model

def recuperar_similares(texto_input, index, df, model, k=5):
  consulta = "Conta gerencial para " + texto_input.lower()
  embed = model.encode([consulta])
  distancias, indices = index.search(np.array(embed), k)
  return df.iloc[indices[0]]
