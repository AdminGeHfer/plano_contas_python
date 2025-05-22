#!/bin/bash

# Diretório do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/streamlit.log"

# Criação da pasta de logs, se não existir
mkdir -p "$LOG_DIR"

# Executa o Streamlit em background com nohup
echo "🚀 Iniciando Streamlit com nohup..."
nohup streamlit run app.py > "$LOG_FILE" 2>&1 &

# Exibe informações para o usuário
echo "✅ Streamlit rodando em segundo plano!"
echo "🌐 Acesse: http://localhost:8501"
echo "📄 Logs sendo salvos em: $LOG_FILE"
echo "📡 Veja os logs ao vivo com: tail -f $LOG_FILE"
echo "🛑 Para parar o Streamlit, use: kill \$(lsof -t -i:8501)"