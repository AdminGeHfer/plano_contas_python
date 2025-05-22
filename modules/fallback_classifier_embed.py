import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer
from modules.utils import normalizar_texto, gerar_embeddings

# Caminhos
CSV_FILE = "data/base_treinamento_fallback.csv"
MODEL_FILE = "models/classificador_fallback_embed.joblib"

# 🔢 Embeddings com modelo multilíngue
modelo_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 🔄 Carregamento dos dados
print("🔄 Carregando base de treinamento…")
df = pd.read_csv(CSV_FILE, delimiter=";", encoding="utf-8")
entradas = df["Descrição do Produto"]
rotulos = df["Conta Gerencial"]

# ⚖️ Rebalanceamento por classe
classes = np.unique(rotulos)
pesos = compute_class_weight(class_weight="balanced", classes=classes, y=rotulos)
pesos_dict = dict(zip(classes, pesos))

# 🔧 Pipeline com embeddings
modelo = Pipeline([
    ("normalize", FunctionTransformer(
      func=normalizar_texto,
      kw_args={"expandir_sinonimos": True, "topn": 3, "limiar": 0.75},
      validate=False
    )),
    ("embed", FunctionTransformer(gerar_embeddings, validate=False)),
    ("clf", LogisticRegression(max_iter=2000, class_weight=pesos_dict))
])

# 🧠 Treinamento
print("⚙️ Treinando modelo com embeddings…")
modelo.fit(entradas, rotulos)

# 💾 Persistência
joblib.dump(modelo, MODEL_FILE, compress=3)
print("✅ Modelo salvo em", MODEL_FILE)
