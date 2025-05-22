import pandas as pd
import streamlit as st
from joblib import load
from dotenv import load_dotenv
from modules.llm_handler import consulta_llm_langchain
from modules.faiss_handler import carregar_index_faiss, recuperar_similares

# ---------------------------- Credenciais LangSmith
load_dotenv()

# ---------------------------- Configuração da página
st.set_page_config(page_title="Assistente Plano de Contas", layout="centered")
st.title("Assistente de Contas Gerenciais")
st.caption("Classificador supervisionado + FAISS + LLM")

# ---------------------------- Carregamentos únicos
index_faiss, df_faiss, _ = carregar_index_faiss()
modelo_fallback = load("models/classificador_fallback_embed.joblib")

# ---------------------------- Interface
consulta = st.text_input("Escreva a situação como na descrição da entrada de NF (ex.: 'CANETA ESFER BIC CRISTAL AVULSO'):")

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

    # 4. Exibir veredicto primeiro
    st.divider()
    st.subheader("🧠 Veredito do LLM")
    st.markdown(resposta_llm)

    # 5. Expanders com demais blocos
    with st.expander("🎯 Conta sugerida (supervisionado)", expanded=False):
        st.write(f"Confiança: {prob_best:.2%}")
        st.markdown(f"**{conta_predita}**")
        df_probs = pd.DataFrame({"Conta": classes, "Prob": proba})
        st.table(df_probs.sort_values("Prob", ascending=False).head(5))

    with st.expander("🔎 FAISS (texto_base)", expanded=False):
        st.markdown(historico_uso or "_Sem histórico encontrado_")

    with st.expander("📊 Métricas da geração", expanded=False):
        st.write(f"Tokens estimados: {token_count}")
