import pandas as pd
import joblib
from modules.fallback_classifier import treinar_classificador

# Caminhos
CSV_FILE = "data/base_treinamento_fallback.csv"
MODEL_FILE = "models/classificador_fallback.joblib"

print("🔄 Treinando classificador supervisionado…")

df = pd.read_csv(CSV_FILE, delimiter=";", encoding="utf-8")
entradas = df["Descrição do Produto"]
rotulos = df["Conta Gerencial"]

modelo = treinar_classificador(entradas, rotulos)
modelo.fit(entradas, rotulos)   # ← 👈 ESSENCIAL

joblib.dump(modelo, MODEL_FILE, compress=3)
print("✅ Modelo salvo em", MODEL_FILE)
