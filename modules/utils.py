import re
import spacy
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
from functools import lru_cache
from tqdm import tqdm

modelo_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
nlp = spacy.load("pt_core_news_lg") # PLN em português

def gerar_embeddings(textos):
  if isinstance(textos, str):
    textos = [textos]
  return modelo_st.encode(textos, show_progress_bar=False).tolist()

@lru_cache(maxsize=512)
def gerar_sinonimos(palavra: str, topn=3, limiar=0.75) -> list[str]:
  doc = nlp(palavra)
  resultados = []

  for lex in nlp.vocab:
    if not lex.is_alpha or lex.is_stop or lex.orth_ == palavra:
      continue

    token_ref = nlp(lex.orth_)
    if doc.vector_norm and token_ref.vector_norm:
      similaridade = doc.similarity(token_ref)
    else:
      continue

    if similaridade >= limiar:
      resultados.append((lex.orth_, similaridade))

  resultados.sort(key=lambda x: -x[1])
  return [w for w, _ in resultados[:topn]]

def limpar_texto(texto: str) -> list[str]:
  texto = unidecode(texto.lower())
  tokens = re.findall(r"\b\w+\b", texto)
  return [t for t in tokens if len(t) > 1]

def normalizar_texto(X, expandir_sinonimos=False, topn=3, limiar=0.75):
  if isinstance(X, str):
    X = [X]

  resultados = []

  for texto in tqdm(X, desc="🔄 Normalizando entradas"):
    tokens = limpar_texto(texto)
    expandido = set(tokens)
    if expandir_sinonimos:
      for token in tokens:
        sinonimos = gerar_sinonimos(token, topn=topn, limiar=limiar)
        expandido.update(sinonimos)

    bigramas = [f"{t1}_{t2}" for t1, t2 in zip(tokens, tokens[1:])]
    resultados.append(" ".join(expandido.union(bigramas)))

  return resultados
