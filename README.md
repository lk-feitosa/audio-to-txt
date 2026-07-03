# Aux-Leg (Gerador de Legendas com Faster-Whisper)

Este projeto utiliza o modelo de inteligência artificial [faster-whisper](https://github.com/SYSTRAN/faster-whisper) para gerar transcrições automáticas de vídeos locais com altíssima precisão e velocidade, utilizando o poder de placas de vídeo NVIDIA (CUDA).

## Pré-requisitos Globais (Sistema)

Para rodar este script em um novo PC, certifique-se de ter os seguintes programas instalados no seu sistema operacional (Linux):

- **Python 3.10+**
- **FFmpeg**: Necessário para extrair e converter o áudio do vídeo antes de mandar para a IA.
  - Fedora: `sudo dnf install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- **Drivers NVIDIA** devidamente instalados no computador.

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

Ao rodar, o terminal pedirá para você arrastar o arquivo de vídeo. O áudio será extraído silenciosamente e a transcrição será salva no formato `[Tempo -> Tempo] Texto` em um arquivo `.txt` ao lado do vídeo original.
