import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from modules.rag_dictionary import consultar_exemplos_similares

llm = OllamaLLM(model="mistral:7b", temperature=0.2)

df_dicionario = pd.read_csv("data/dicionario_uso_contas.csv", delimiter=";", encoding="utf-8")

prompt = PromptTemplate(
  input_variables=["input_usuario", "resposta_fallback", "historico_uso"],
  template="""
  Você é analista contábil.  
  Sua missão:

  1. Verifique **se a conta sugerida** pelo classificador é a melhor opção.
  2. Se concordar, responda:  OK  - <Conta Gerencial>  - <1-linha de justificativa>.
  3. Se existir outra conta **mais apropriada**, responda: CORRIGIR  - <Conta Melhor>  - <1-linha de justificativa>.

  Pergunta: "{input_usuario}"  
  Conta sugerida: "{resposta_fallback}"  
  Exemplos semelhantes encontrados: "{historico_uso}"
  """
)

chain = prompt | llm

def consulta_llm_langchain(input_usuario, historico_uso, resposta_fallback):
  # Ignora candidatos_df (por ora não usado neste modo)
  exemplos_semelhantes = consultar_exemplos_similares(input_usuario, resposta_fallback, k=5)

  if not exemplos_semelhantes:
    historico_uso = "Sem histórico encontrado."
  else:
    historico_uso = "\n".join(f"- {x}" for x in exemplos_semelhantes)

  return chain.invoke({
    "input_usuario": input_usuario,
    "resposta_fallback": resposta_fallback,
    "historico_uso": historico_uso
  })
