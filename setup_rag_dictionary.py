import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Entradas e saídas
ENTRADA_CSV = "data/dicionario_uso_contas.csv"
SAIDA_CSV = "data/dicionario_embeddings.csv"
FAISS_INDEX = "data/dicionario_faiss.index"

# Carregar dicionário agrupado
df_dicionario = pd.read_csv(ENTRADA_CSV, delimiter=";")

# Expandir lista de exemplos
expanded_rows = []
for _, row in df_dicionario.iterrows():
    conta = row["Conta Gerencial"]
    try:
        exemplos = eval(row["Exemplos de Uso"]) if isinstance(row["Exemplos de Uso"], str) else row["Exemplos de Uso"]
        for exemplo in exemplos:
            expanded_rows.append({"Conta Gerencial": conta, "Exemplo de Uso": exemplo})
    except Exception as e:
        print(f"[ERRO] Conta {conta} - {e}")

df_expandidos = pd.DataFrame(expanded_rows)

# Gerar embeddings
print("🔄 Gerando embeddings...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
corpus = df_expandidos["Exemplo de Uso"].astype(str).tolist()
embeddings = model.encode(corpus, show_progress_bar=True)

# Criar índice FAISS
print("📦 Criando índice FAISS...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Salvar base expandida e índice
print("💾 Salvando arquivos...")
df_expandidos.to_csv(SAIDA_CSV, index=False, sep=";")
faiss.write_index(index, FAISS_INDEX)

print("✅ Dicionário vetorizado gerado com sucesso.")
