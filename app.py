# app.py
import streamlit as st
import pandas as pd
from joblib import load

from modules.faiss_handler import carregar_index_faiss, recuperar_similares         # (continua igual)
from modules.rag_dictionary import consultar_exemplos_similares                    # nova versão com busca global
from modules.llm_handler import chain                                              # só o chain – o handler agora está dentro.

# ---------------------------- Configuração de página
st.set_page_config(page_title="Assistente Plano de Contas", layout="centered")
st.title("Assistente de Contas Gerenciais")
st.caption("Classificador supervisionado + RAG + LLM")

# ---------------------------- Carregamentos únicos
#
# 1) Índice FAISS dos planos de conta (campo texto_base) – aparece na “visualização” mas não participa mais da validação
index_faiss, df_faiss, _ = carregar_index_faiss()

# 2) Classificador supervisionado treinado/serializado
modelo_fallback = load("models/classificador_fallback.joblib")

# 3) (Opcional) dicionário em memória p/ debug
df_dicionario = pd.read_csv("data/dicionario_embeddings.csv", delimiter=";")

# ---------------------------- UI principal
consulta = st.text_input("Descreva a situação (ex.: “compra de lápis”):")

if consulta:
    # ---------- 1) Predição supervisionada
    proba = modelo_fallback.predict_proba([consulta])[0]
    conf_max = max(proba)
    classes = modelo_fallback.classes_
    best_idx = proba.argmax()
    conta_predita = classes[best_idx]
    prob_best     = proba[best_idx]

    # ---------- 2) RAG (dicionário de usos) usando a conta predita COMO PREFERIDA
    exemplos = consultar_exemplos_similares(
                  texto_usuario=consulta,
                  conta_sugerida=conta_predita,
                  k=5,                 # quantos exemplos mandar ao prompt
                  busca_global_k=50    # quão “largo” buscar no corpus
              )

    # ---------- 3) Chamada ao LLM
    historico_uso = "\n".join(f"- {ex}" for ex in exemplos)
    prompt_vars = {
        "input_usuario":   consulta,
        "resposta_fallback": conta_predita,
        "historico_uso": historico_uso,
    }
    resposta_llm = chain.invoke(prompt_vars)

    # ---------- 4) Interface
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Conta sugerida (supervisionado)")
        st.write(f"Confiança do classificador supervisionado: {conf_max:.2%}")
        st.markdown(f"**{conta_predita}**  \n`probabilidade: {prob_best:.1%}`")

        with st.expander("Top-5 probabilidades"):
            df_probs = (
                pd.DataFrame({"Conta": classes, "Prob": proba})
                  .sort_values("Prob", ascending=False)
                  .head(5)
            )
            st.table(df_probs)

    with col2:
        st.subheader("🔎 Exemplos mais parecidos (RAG)")
        st.markdown(historico_uso or "_Sem histórico encontrado_")

    st.divider()
    st.subheader("🧠 Veredicto do LLM")
    st.markdown(resposta_llm.strip())

    # ---------- 5) (Debug opcional) vetores FAISS – mostre o texto_base dos 3 mais próximos
    with st.expander("Debug FAISS (texto_base corporativo)"):
        faiss_hits = recuperar_similares(consulta, index_faiss, df_faiss, _)
        st.table(faiss_hits[["PLANO", "GRUPO", "texto_base"]].head(3))
