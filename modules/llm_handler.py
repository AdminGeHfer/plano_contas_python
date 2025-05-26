import os
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI # from langchain_ollama import OllamaLLM (for ollama models)
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from unidecode import unidecode
from dotenv import load_dotenv

load_dotenv()
tracer = LangChainTracer()
llm = ChatOpenAI(
  model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125"),
  openai_api_key=os.getenv("OPENAI_API_KEY"),
  openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
  temperature=0.2,
  max_tokens=600,
) # llm = OllamaLLM(model="mistral:7b", temperature=0.2) # for ollama models
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", legacy=False)

prompt = PromptTemplate(
  input_variables=["input_usuario", "candidatos", "textos_base"],
  template="""
  Você é um assistente de conta gerenciais. Sua tarefa é avaliar cinco sugestões de contas gerenciais retornadas por um classificador supervisionado e compará-las com três exemplos reais da base contábil institucional (texto_base).

  Solicitação do usuário: {input_usuario}

  Sugestões com pontuação:
  {candidatos}

  Exemplos da base real:
  {textos_base}

  Com base nas sugestões e exemplos reais, indique a conta mais apropriada.

  Como prioridade, considere as probabilidades das sugestões com pontuação.

  Caso nenhuma delas seja adequada (pode até considerar a probabilidade como parâmetro), leve em consideração os exemplos da base real.

  Seja direto e justifique com base textual clara. Destaque a conta escolhida em negrito.
  """
)

chain = prompt | llm

def consulta_llm_langchain(input_usuario, contas_top5, scores_top5, textos_base):
    candidatos_str = "\n".join([
        f"- {conta} (probabilidade: {round(score*100, 2)}%)"
        for conta, score in zip(contas_top5, scores_top5)
    ])

    textos_str = "\n".join(f"- {t}" for t in textos_base)

    full_prompt = prompt.format(
        input_usuario=unidecode(input_usuario.upper()),
        candidatos=candidatos_str,
        textos_base=textos_str
    )

    token_count = len(tokenizer.encode(full_prompt))

    resposta = chain.invoke({
        "input_usuario": input_usuario,
        "candidatos": candidatos_str,
        "textos_base": textos_str
    }) # , config={"callbacks": [tracer]})

    return resposta, token_count # .strip()
