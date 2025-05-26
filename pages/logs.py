import gradio as gr
from joblib import load

def carregar_logs():
  try:
    with open("logs/atualizacoes.log", "r") as f:
      return f.read()
  except FileNotFoundError:
    return "⚠️ Log de atualizações ainda não foi gerado."

def checar_modelo():
  try:
    modelo = load("models/classificador_fallback_embed.joblib")
    if hasattr(modelo, 'steps'):
      steps = "\n".join([f"- {nome}: {type(etapa).__name__}" for nome, etapa in modelo.steps])
      return f"✅ Modelo carregado com sucesso.\n\n### Pipeline:\n{steps}"
    else:
      return "⚠️ O modelo carregado não é um pipeline sklearn."
  except Exception as e:
    return f"❌ Erro ao carregar o modelo: {e}"

def interface_logs():
  gr.Markdown("## 📜 Logs do Sistema")

  with gr.Row():
    gr.Textbox(value=carregar_logs(), lines=15, label="📅 Histórico de atualizações")

  with gr.Row():
    gr.Textbox(value=checar_modelo(), lines=10, label="🔍 Diagnóstico do modelo")
