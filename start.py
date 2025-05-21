import os
import hashlib
import time
import subprocess
from datetime import datetime

# Arquivos monitorados
DATA_PATH = "data/base_treinamento_fallback.csv"
HASH_PATH = "data/hash_base.md5"
LOG_PATH = "logs/atualizacoes.log"

# Scripts de build
CLASSIFIER_BUILD = "modules.fallback_classifier_embed"
RAG_BUILD       = "setup_rag_dictionary"

# Utilitários
def gerar_hash(arquivo):
    hasher = hashlib.md5()
    with open(arquivo, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def carregar_hash_antigo():
    if not os.path.exists(HASH_PATH):
        return None
    with open(HASH_PATH, "r") as f:
        return f.read().strip()

def salvar_novo_hash(hash_str):
    with open(HASH_PATH, "w") as f:
        f.write(hash_str)

def log(msg):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def executar_script(script_path):
    print(f"⚙️ Executando: {script_path}")
    start = time.time()
    result = subprocess.run(["python", "-m", script_path])
    tempo = time.time() - start
    print(f"✅ Concluído ({tempo:.2f} segundos)\n")
    return result.returncode == 0

# --------------------------
# Execução principal
# --------------------------
print("🔍 Verificando modificações na base...")

if not os.path.exists(DATA_PATH):
    print(f"❌ Arquivo não encontrado: {DATA_PATH}")
    exit(1)

hash_atual = gerar_hash(DATA_PATH)
hash_antigo = carregar_hash_antigo()

if hash_atual != hash_antigo:
    print("📊 Alterações detectadas na base de treinamento!")
    log("Alterações detectadas - regenerando modelos...")

    sucesso1 = executar_script(CLASSIFIER_BUILD)
    sucesso2 = executar_script(RAG_BUILD)

    if sucesso1 and sucesso2:
        salvar_novo_hash(hash_atual)
        log("Modelos atualizados com sucesso.")
        print("🚀 Base processada com sucesso.")
    else:
        log("Erro durante atualização dos modelos.")
        print("❌ Ocorreu um erro ao atualizar os modelos.")
else:
    print("✅ Nenhuma modificação detectada.")
    log("Nenhuma alteração - modelos intactos.")
