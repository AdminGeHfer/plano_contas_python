# LLM: Assistente plano de contas
O intuito deste projeto é construir um modelo de LLM local no qual realiza a análise da descrição do item que o usuário está com dúvidas de qual conta gerencial usar, então o LLM com o apoio dos conceitos de aprendizado de máquina supervisionado e PLN (Processamento de Linguagem Natural), a construção do assistente nessa primeira versão tem as seguintes funcionalidades:

- Aba principal: Assistente de Contas Gerenciais
  - Esta aba contêm a funcionalidade principal deste projeto, em que ao usuário descrever uma situação como na descrição da entrada de uma Nota Fiscal, o LLM irá consultar o modelo de aprendizado de máquina supervisionado e o banco de dados vetorial, para a partir disso, tomar uma conclusão sobre qual conta gerencial é a mais adequada para o cenário descrito, levando em consideração as duas análises realizadas.
  - Além disso, é apresentado após o parecer do LLM, as top 5 escolhas de contas sugeridas do aprendizado supervisionado, assim como as top 3 escolhas do banco de dados vetorial, e por fim a quantidade de tokens que foi consumida/gerada pelo LLM para a resposta exibida.
- Aba secundária: Avaliação do Modelo Supervisionado
  - Afim de ter uma análise manual sobre a acurácia dos resultados do modelo supervisionado, o projeto conta com o apoio de PLN com treinamento divido em arquivos para analisar todos os registros históricos, que estão na base de treinamento, com duas opções: normalização sem sinônimos e com sinônimos. Dessa forma, após o treinamento/processamento, a avaliação retorna a acurácia do top-1, a acurácia do top-5, e retorna uma tabela com todas as predições de forma individual, no qual apresenta a descrição, a conta gerencial verdadeira e a conta gerencial predita.
- Aba secundária: Logs
  - Por fim, a aba de logs está presente apenas para poder monitorar caso tenha alguma alteração nos modelos, ou relacionado a parte de atualização e regeneração dos modelos, para poder assim ver o log completo pelo console, e ter uma visão um pouco mais superficial.

## Ajustes e melhorias
Conforme as necessidades de demandas e melhorias necessárias do assistente, é necessário levar em conta as seguintes considerações:

- Caso haja alguma atualização na base de dados histórica, no caso o `base_treinamento_fallback.csv`, será necessário refazer o treinamento da avaliação do PLN do modelo supervisionado (com e sem sinônimos), por meio da aba `avaliacao` do Gradio.
  - Lembrando que por questão de otimização de tempo, do modelo supervisionado com sinônimos leva mais tempo que o sem sinônimos (com ≈ 4h; sem ≈ 5min).

- Caso ocorra alteração no `dataset_plano_de_contas.csv` ou no `dicionario_uso_contas.csv`, é necessário re-executar o setup do RAG junto com o FAISS, no qual para isso basta executar no terminal:

```python
python start.py
```

### Ferramentas e configurações
O projeto atualmente utiliza as seguintes bibliotecas e ferramentas:
- Python (versão atual: 3.10.14)
- Modelo de LLM: [`mistral:7b`](https://ollama.com/library/mistral:7b) com temperatura 0.2 via Ollama (se for usado localmente)
- Modelo de LLM (GPT): [`gpt-3.5-turbo`](https://platform.openai.com/docs/models/gpt-3.5-turbo)
- Modelo de tokenização do LLM: [`hf-internal-testing/llama-tokenizer`](https://huggingface.co/hf-internal-testing/llama-tokenizer)
- Modelo de PLN (Baixado via spaCy): [`pt_core_news_lg`](https://spacy.io/models/pt)
- Modelo de Deep Learning (significado semântico de sentenças): [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Bibliotecas utilizadas:
  - Nativas:
    - [OS](https://docs.python.org/3/library/os.html)
    - [HashLib](https://docs.python.org/3/library/hashlib.html)
    - [Time](https://docs.python.org/3/library/time.html)
    - [Subprocess](https://docs.python.org/3/library/subprocess.html)
    - [Datetime](https://docs.python.org/3/library/datetime.html)
    - [Pathlib](https://docs.python.org/3/library/pathlib.html)
    - [RE](https://docs.python.org/3/library/re.html)
    - [Functools](https://docs.python.org/3/library/functools.html)
  - [Pandas](https://pandas.pydata.org)
  - [Numpy](https://numpy.org)
  - [FAISS](https://ai.meta.com/tools/faiss)
  - [SentenceTransformers](https://sbert.net)
  - [Joblib](https://joblib.readthedocs.io/en/stable)
  - [Dotenv](https://github.com/theskumar/python-dotenv)
  - [Scikit-Learn](https://scikit-learn.org/stable)
  - [Spacy](https://spacy.io)
  - [Unidecode](https://github.com/avian2/unidecode)
  - [Tqdm](https://tqdm.github.io)
  - [Transformers](https://github.com/huggingface/transformers)
  - [LangchainOllama](https://python.langchain.com/docs/integrations/chat/ollama)
  - [Langchain](https://www.langchain.com)
  - [Gradio](https://www.gradio.app)

- Ferramenta para visualização das informações: [Gradio](https://www.gradio.app)
- Uso do Ollama localmente (e baixar o modelo usado):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b
```
- Instalação do modelo de PLN via spacy (Python):
```python
python -m spacy download pt_core_news_lg
```
- [PyEnv](https://github.com/pyenv/pyenv) para gerenciar a versão do Python.
- Instalação com pyenv:
```bash
pyenv install 3.10.14
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Subir o Gradio:
```bash
nohup python app.py
```
