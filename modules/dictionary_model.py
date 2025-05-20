import pandas as pd

df = pd.read_csv("data/base_treinamento_fallback.csv", delimiter=";", encoding="utf-8")

# Agrupar descrições por conta
agrupado = df.groupby("Conta Gerencial")["Descrição do Produto"].apply(list).reset_index()

# Limitar a 50 exemplos por conta para resumo
agrupado["Exemplos de Uso"] = agrupado["Descrição do Produto"].apply(lambda lista: lista[:50])
agrupado.drop(columns=["Descrição do Produto"], inplace=True)

# Salvar como dicionário de uso
agrupado.to_csv("data/dicionario_uso_contas.csv", index=False, sep=";")
