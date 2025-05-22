import streamlit as st
import pandas as pd
import os
from joblib import dump, load
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from modules.utils import normalizar_texto
from joblib import load
from sklearn.pipeline import Pipeline

# -------------------- Configurações fixas
CSV_PATH = "data/base_treinamento_fallback.csv"
MODEL_PATH = "models/classificador_fallback_embed.joblib"
BATCH_SIZE = 500

# -------------------- Interface: escolha da configuração
st.subheader("📊 Avaliação do Modelo Supervisionado")

modo = st.radio("Escolha a forma de normalização:", ["Sem sinônimos", "Com sinônimos"], horizontal=True)

config = {
  "expandir_sinonimos": modo == "Com sinônimos",
  "cache_dir": "cache_avaliacao_com_sinonimos" if modo == "Com sinônimos" else "cache_avaliacao_sem_sinonimos"
}
os.makedirs(config["cache_dir"], exist_ok=True)

# -------------------- Carregamento
with st.expander("🔍 Resultados de Acurácia (Top-1 e Top-5)", expanded=True):
  modelo = load(MODEL_PATH)
  df = pd.read_csv(CSV_PATH, delimiter=";", encoding="utf-8")
  X_raw = df["Descrição do Produto"].astype(str).tolist()
  y_true = df["Conta Gerencial"].astype(str).tolist()

  st.info(f"Avaliando em batches de {BATCH_SIZE} itens... (modo: {modo})")

  y_preds = []
  y_probas = []

  for i in range(0, len(X_raw), BATCH_SIZE):
    batch_X = X_raw[i:i+BATCH_SIZE]
    cache_pred = os.path.join(config["cache_dir"], f"pred_{i}.joblib")
    cache_proba = os.path.join(config["cache_dir"], f"proba_{i}.joblib")

    if os.path.exists(cache_pred) and os.path.exists(cache_proba):
      st.write(f"✔️ Batch {i}-{i+len(batch_X)}: cache encontrado.")
      y_batch = load(cache_pred)
      p_batch = load(cache_proba)
    else:
      st.write(f"🔄 Processando batch {i}-{i+len(batch_X)}...")
      X_norm = normalizar_texto(batch_X, expandir_sinonimos=config["expandir_sinonimos"])
      y_batch = modelo.predict(X_norm)
      p_batch = modelo.predict_proba(X_norm)
      dump(y_batch, cache_pred)
      dump(p_batch, cache_proba)

    y_preds.extend(y_batch)
    y_probas.extend(p_batch)

  # -------------------- Métricas finais
  st.success("✅ Avaliação concluída!")
  acc_top1 = accuracy_score(y_true, y_preds)
  acc_top5 = top_k_accuracy_score(y_true, y_probas, k=5, labels=modelo.classes_)

  st.write(f"🎯 **Acurácia Top-1**: {acc_top1:.2%}")
  st.write(f"📌 **Acurácia Top-5**: {acc_top5:.2%}")

  if st.checkbox("🔍 Visualizar predições individuais"):
    df_preds = pd.DataFrame({
      "Descrição": X_raw,
      "Verdadeira": y_true,
      "Predita": y_preds
    })
    st.dataframe(df_preds)
