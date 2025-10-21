import gradio as gr
import pandas as pd
import os
from joblib import load, dump
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from modules.utils import normalizar_texto

CSV_PATH = "data/base_treinamento_fallback.csv"
MODEL_PATH = "models/classificador_fallback_embed.joblib"
BATCH_SIZE = 500

def avaliar_modelo(modo):
  expandir = modo == "Com sinônimos"
  cache_dir = "cache_avaliacao_com_sinonimos" if expandir else "cache_avaliacao_sem_sinonimos"
  os.makedirs(cache_dir, exist_ok=True)

  modelo = load(MODEL_PATH)
  df = pd.read_csv(CSV_PATH, delimiter=";", encoding="utf-8")
  X_raw = df["DESCRI"].astype(str).tolist()
  y_true = df["Conta Gerencial"].astype(str).tolist()

  y_preds = []
  y_probas = []

  for i in range(0, len(X_raw), BATCH_SIZE):
    batch_X = X_raw[i:i+BATCH_SIZE]
    cache_pred = os.path.join(cache_dir, f"pred_{i}.joblib")
    cache_proba = os.path.join(cache_dir, f"proba_{i}.joblib")

    if os.path.exists(cache_pred) and os.path.exists(cache_proba):
      y_batch = load(cache_pred)
      p_batch = load(cache_proba)
    else:
      X_norm = normalizar_texto(batch_X, expandir_sinonimos=expandir)
      y_batch = modelo.predict(X_norm)
      p_batch = modelo.predict_proba(X_norm)
      dump(y_batch, cache_pred)
      dump(p_batch, cache_proba)

    y_preds.extend(y_batch)
    y_probas.extend(p_batch)

  acc_top1 = accuracy_score(y_true, y_preds)
  acc_top5 = top_k_accuracy_score(y_true, y_probas, k=5, labels=modelo.classes_)

  df_preds = pd.DataFrame({
    "Descrição": X_raw,
    "Verdadeira": y_true,
    "Predita": y_preds
  })

  return f"{acc_top1:.2%}", f"{acc_top5:.2%}", df_preds

def interface_avaliacao():
  with gr.Column():
    gr.Markdown("## 📊 Avaliação do Modelo Supervisionado")

    modo = gr.Radio(["Sem sinônimos", "Com sinônimos"], label="Forma de normalização", value="Sem sinônimos")
    acc1 = gr.Textbox(label="🎯 Acurácia Top-1")
    acc5 = gr.Textbox(label="📌 Acurácia Top-5")

    tabela = gr.Dataframe(
      label="🔍 Amostra de predições",
      interactive=True,
      row_count="dynamic",
      wrap=True
    )

    botao = gr.Button("Executar Avaliação")
    botao.click(fn=avaliar_modelo, inputs=modo, outputs=[acc1, acc5, tabela])
