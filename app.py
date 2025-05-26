import gradio as gr
from pages.assistente import interface_assistente
from pages.avaliacao import interface_avaliacao
from pages.logs import interface_logs

with gr.Blocks(title="Assistente de Contas Gerenciais") as demo:
  with gr.Tabs():
    with gr.Tab("Assistente"):
      interface_assistente()
    with gr.Tab("Avaliação"):
      interface_avaliacao()
    with gr.Tab("Logs"):
      interface_logs()

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=7860, share=False)  # Use share=True for public access
