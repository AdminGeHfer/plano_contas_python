import pandas as pd
import streamlit as st
from joblib import load
from dotenv import load_dotenv
from modules.llm_handler import consulta_llm_langchain
from modules.faiss_handler import carregar_index_faiss, recuperar_similares

load_dotenv()

st.set_page_config(page_title="Assistente Plano de Contas", layout="centered", page_icon="🧾")
st.title("Assistente de Contas Gerenciais")
st.caption("Classificador supervisionado + FAISS + LLM")

index_faiss, df_faiss, _ = carregar_index_faiss()
modelo_fallback = load("models/classificador_fallback_embed.joblib")

consulta = st.text_input("Escreva o nome do produto como é exibido na descrição da Nota Fiscal (ex.: 'CANETA ESFER BIC CRISTAL AVULSO'):")

if consulta:
  # 1. Predição supervisionada
  proba = modelo_fallback.predict_proba([consulta])[0]
  classes = modelo_fallback.classes_
  best_idx = proba.argmax()
  conta_predita = classes[best_idx]
  prob_best = proba[best_idx]

  # 2. RAG com top-3 texto_base (FAISS)
  faiss_hits = recuperar_similares(consulta, index_faiss, df_faiss, _)
  top3_textos = faiss_hits["texto_base"].head(3).tolist()
  historico_uso = "\n".join(f"- {x}" for x in top3_textos)

  # 3. LLM com top-5 supervisionado + texto_base real
  top5_idx = proba.argsort()[-5:][::-1]
  top5_contas = classes[top5_idx]
  top5_scores = proba[top5_idx]

  resposta_llm, token_count = consulta_llm_langchain(
    input_usuario=consulta,
    contas_top5=top5_contas,
    scores_top5=top5_scores,
    textos_base=top3_textos
  )

  st.divider()
  st.subheader("🧠 Veredito do LLM")
  st.markdown(resposta_llm)

  with st.expander("🎯 Conta sugerida (supervisionado)", expanded=False):
    st.write(f"Confiança: {prob_best:.2%}")
    st.markdown(f"**{conta_predita}**")
    df_probs = pd.DataFrame({"Conta": classes, "Probabilidade": proba})
    df_probs["Probabilidade"] = (df_probs["Probabilidade"] * 100).apply(lambda x: f"{x:.2f}%")
    st.table(df_probs.sort_values("Probabilidade", ascending=False).head(5))

  with st.expander("🔎 FAISS (vector database)", expanded=False):
    st.markdown(historico_uso or "_Sem histórico encontrado_")

  with st.expander("📊 Métricas da geração (tokens)", expanded=False):
    st.write(f"Tokens estimados: {token_count}")
