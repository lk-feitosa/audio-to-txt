#!/bin/bash
set -e

echo "========================================="
echo "   PREPARANDO AMBIENTE COM 'uv' 🚀"
echo "========================================="

# Verifica se o uv está instalado, se não estiver, instala
if ! command -v uv &> /dev/null; then
    echo "⚠️ O 'uv' não foi encontrado."
    echo "Instalando o 'uv' (gerenciador ultrarrápido)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Adiciona o uv ao path da sessão atual para podermos usar logo em seguida
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# Cria o ambiente virtual e instala as dependências em uma velocidade absurda
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual..."
    uv venv
fi

echo "⚡ Sincronizando dependências (isso é extremamente rápido com uv)..."
uv pip install -r requirements.txt

echo "✅ Ambiente pronto!"
echo "Iniciando a aplicação..."
echo ""

# Executa o main.py usando o ambiente virtual gerenciado pelo uv
uv run python main.py
