import pandas as pd

TREINO = "data/base_treinamento_fallback.csv"
PLANO  = "data/dataset_plano_de_contas.csv"   # seu arquivo mais recente

df_treino = pd.read_csv(TREINO,  delimiter=";")
df_plano  = pd.read_csv(PLANO,   delimiter=";")   # ajuste delimitador se necessário

# todas as contas no plano
contas_plano = set(df_plano["DESC_PLA"].str.strip())

# contas já cobertas pelo treino
contas_treino = set(df_treino["Conta Gerencial"].str.strip())

faltantes = sorted(contas_plano - contas_treino)
print(f"Contas novas sem exemplos ({len(faltantes)}):")
for c in faltantes:
  print(" -", c)
