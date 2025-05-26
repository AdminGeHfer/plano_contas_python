import gradio as gr
import pandas as pd
from joblib import load
from modules.llm_handler import consulta_llm_langchain
from modules.faiss_handler import carregar_index_faiss, recuperar_similares

index_faiss, df_faiss, _ = carregar_index_faiss()
modelo_fallback = load("models/classificador_fallback_embed.joblib")

def processar(consulta):
  if not consulta.strip():
    return "Insira uma descrição válida.", None, None, None

  proba = modelo_fallback.predict_proba([consulta])[0]
  classes = modelo_fallback.classes_
  best_idx = proba.argmax()
  conta_predita = classes[best_idx]
  prob_best = proba[best_idx]

  faiss_hits = recuperar_similares(consulta, index_faiss, df_faiss, _)
  top3_textos = faiss_hits["texto_base"].head(3).tolist()
  historico_uso = "\n".join(f"- {x}" for x in top3_textos)

  top5_idx = proba.argsort()[-5:][::-1]
  top5_contas = classes[top5_idx]
  top5_scores = proba[top5_idx]

  resposta_llm, token_count = consulta_llm_langchain(
    input_usuario=consulta,
    contas_top5=top5_contas,
    scores_top5=top5_scores,
    textos_base=top3_textos
  )

  top5_df = pd.DataFrame({
    "Conta": top5_contas,
    "Probabilidade": [f"{p*100:.2f}%" for p in top5_scores]
  })

  text = getattr(resposta_llm, "content", str(resposta_llm))
  return text.strip(), conta_predita, historico_uso, top5_df

def interface_assistente():
  gr.Markdown("## 🧾 Assistente de Contas Gerenciais")

  with gr.Row():
    entrada = gr.Textbox(label="Escreva o nome do produto como é exibido na descrição da Nota Fiscal", placeholder="Ex.: CANETA ESFER BIC CRISTAL AVULSO")
  with gr.Row():
    saida_llm = gr.Textbox(label="🧠 Veredito do LLM")
    conta_supervisionado = gr.Textbox(label="🎯 Conta Sugerida")
  with gr.Row():
    textos_base = gr.Textbox(label="🔎 Exemplos do FAISS")
    tabela_top5 = gr.Dataframe(headers=["Conta", "Probabilidade"], label="📊 Top-5 Contas")

  botao = gr.Button("Executar")
  botao.click(fn=processar, inputs=entrada, outputs=[saida_llm, conta_supervisionado, textos_base, tabela_top5])
