import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Logs do Sistema", layout="centered")
st.title("📑 Logs do Sistema")
st.caption("Inspeção dos registros de atualizações e atividades")

LOG_PATH = Path("logs/atualizacoes.log")

if LOG_PATH.exists():
    with LOG_PATH.open("r", encoding="utf-8") as f:
        linhas = f.readlines()

    if linhas:
        st.info(f"{len(linhas)} registros encontrados.")
        st.code("".join(reversed(linhas)), language="text")
    else:
        st.warning("O arquivo de log existe, mas está vazio.")
else:
    st.error("Arquivo de log `logs/atualizacoes.log` não encontrado.")
    st.markdown("⚠️ Verifique se o script `start.py` está sendo executado para gerar os registros.")
