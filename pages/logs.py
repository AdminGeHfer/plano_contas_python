import streamlit as st
from joblib import load

st.set_page_config(page_title="Logs do Sistema", layout="centered", page_icon="📜")
st.title("📜 Logs e Verificações")

# Exibição do log de atualizações
st.subheader("📅 Histórico de atualizações")
try:
  with open("logs/atualizacoes.log", "r") as f:
    conteudo = f.read()
    st.text_area("Log de Atualizações", value=conteudo, height=300)
except FileNotFoundError:
  st.warning("Log de atualizações ainda não foi gerado.")

# Validação do modelo carregado
st.subheader("🔍 Diagnóstico do modelo supervisionado")
try:
  modelo = load("models/classificador_fallback_embed.joblib")
  st.success("✅ Modelo carregado com sucesso.")

  if hasattr(modelo, 'steps'):
    st.markdown("### Pipeline carregado:")
    for nome, etapa in modelo.steps:
      st.write(f"- **{nome}**: `{type(etapa).__name__}`")
  else:
    st.warning("O modelo carregado não é um pipeline sklearn. Verifique se está correto.")
except Exception as e:
  st.error(f"❌ Erro ao carregar o modelo: {e}")
