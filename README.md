# Aux-Leg (Gerador de Legendas com Faster-Whisper)

Este projeto utiliza o modelo de inteligência artificial [faster-whisper](https://github.com/SYSTRAN/faster-whisper) para gerar transcrições automáticas de vídeos locais com altíssima precisão e velocidade, utilizando o poder de placas de vídeo NVIDIA (CUDA).

## Pré-requisitos Globais (Sistema)

Para rodar este script em um novo PC, certifique-se de ter os seguintes programas instalados no seu sistema operacional (Linux):

- **Python 3.10+**
- **FFmpeg**: Necessário para extrair e converter o áudio do vídeo antes de mandar para a IA.
  - Fedora: `sudo dnf install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- **Placa de Vídeo NVIDIA (Opcional, mas recomendado)**: O script possui um sistema inteligente de *fallback*. Se você tiver uma NVIDIA, ele usará a GPU para transcrições super rápidas. Se você usar AMD, Intel ou não tiver GPU, ele processará automaticamente usando a CPU da máquina.

## 🚀 Instalação e Execução

Este projeto utiliza o **`uv`** para gerenciar o ambiente e as dependências de forma ultrarrápida. Para configurar o projeto (baixar dependências) e rodá-lo, execute os comandos:

```bash
chmod +x run.sh
./run.sh
```

O script cuidará de tudo automaticamente:
1. Instalação do `uv` (se necessário)
2. Criação do ambiente virtual (`.venv`)
3. Instalação super-rápida das dependências (incluindo as bibliotecas CUDA)
4. Execução do `main.py`

Ao rodar, o terminal pedirá para você arrastar o arquivo de vídeo. Logo em seguida, você poderá escolher o idioma da transcrição de forma interativa. 

O áudio será extraído silenciosamente pelo FFmpeg e a transcrição será salva no formato `[Tempo -> Tempo] Texto` em um arquivo `.txt` ao lado do vídeo original.

## ✨ Recursos Inteligentes Inclusos

- **Fallback Automático (NVIDIA vs AMD/CPU):** O script tenta utilizar a aceleração CUDA. Se falhar por incompatibilidade de hardware (ex: Placas AMD), ele recua de forma invisível para o processamento via processador (CPU), garantindo que o programa rode em qualquer computador.
- **Múltiplas Legendas Simultâneas:** O script permite gerar várias legendas (ex: gerar um arquivo em Português e outro em Inglês) em uma única execução.
- **Modo Áudio Misto (Tradução Mágica):** Em vídeos onde as pessoas misturam idiomas constantemente (code-switching), a IA tradicional costuma "alucinar" pulando trechos ou travando. Para resolver isso, o modo Áudio Misto extrai o áudio bruto de forma nativa e utiliza a API do **Google Translator** para traduzir o texto com perfeição (e sem erros) para quantos idiomas você pedir, simultaneamente!
