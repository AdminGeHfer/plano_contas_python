#!/bin/bash

mkdir -p logs

echo "🚀 Iniciando Gradio com nohup..."
nohup python app.py > logs/gradio.log 2>&1 &

echo "✅ Aplicação rodando em segundo plano!"
echo "📡 Veja os logs com: tail -f logs/gradio.log"
echo "🛑 Para encerrar: kill \$(lsof -t -i:7860)"
