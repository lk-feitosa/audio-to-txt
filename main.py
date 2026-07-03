import datetime
import subprocess
import os
import ctypes

# Tenta carregar as bibliotecas CUDA do Python automaticamente (Resolve erro libcublas.so.12)
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    
    # Usa __path__[0] porque em algumas versões do python o __file__ pode ser None
    cublas_dir = nvidia.cublas.lib.__path__[0]
    cudnn_dir = nvidia.cudnn.lib.__path__[0]
    
    # Carrega as bibliotecas na memória para que o ctranslate2 consiga achá-las
    ctypes.cdll.LoadLibrary(os.path.join(cublas_dir, "libcublas.so.12"))
    ctypes.cdll.LoadLibrary(os.path.join(cudnn_dir, "libcudnn.so.9")) # Pode ser so.8 dependendo da versão
except Exception as e:
    pass

from faster_whisper import WhisperModel
from tqdm import tqdm

print("=========================================")
print("🎙️  GERADOR DE LEGENDA (FASTER-WHISPER)  🎙️")
print("=========================================")

# 1. Solicita o arquivo de vídeo ao usuário
raw_input = input("\nArraste o arquivo de vídeo para este terminal ou digite o caminho dele:\n> ").strip()

arquivo_video = raw_input.strip("'").strip('"')

# Tratamento para quando o terminal (como o Zsh) escapa caracteres ao arrastar o arquivo
if not os.path.exists(arquivo_video):
    import shlex
    try:
        parsed = shlex.split(raw_input)
        if parsed and os.path.exists(parsed[0]):
            arquivo_video = parsed[0]
    except Exception:
        pass

# Fallback manual para escapes simples (Zsh)
if not os.path.exists(arquivo_video):
    arquivo_video = raw_input.replace("\\'", "'").replace('\\"', '"').replace("\\ ", " ").strip("'").strip('"')

if not os.path.exists(arquivo_video):
    print(f"\n❌ Erro: O arquivo não foi encontrado.")
    print(f"Caminho interpretado: {arquivo_video}")
    print("Dica: Tente colar o caminho sem aspas e sem barras invertidas (\\).")
    exit(1)

# Configurações do arquivo
arquivo_audio = "audio_otimizado.mp3"
# Salva o arquivo de texto com o mesmo nome do vídeo (mas com .txt)
arquivo_saida = f"{os.path.splitext(arquivo_video)[0]}_transcricao.txt"

# 2. Extrai e otimiza o áudio usando FFmpeg
print(f"\n⚙️  Extraindo e otimizando o áudio de: {os.path.basename(arquivo_video)}...")
comando_ffmpeg = [
    "ffmpeg", "-y", "-i", arquivo_video, 
    "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", 
    arquivo_audio
]

try:
    # Executa o ffmpeg em segundo plano silenciosamente
    subprocess.run(comando_ffmpeg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Áudio otimizado com sucesso!")
except FileNotFoundError:
    print("\n❌ Erro: O programa 'ffmpeg' não está instalado no seu sistema.")
    print("Instale-o no terminal com: sudo dnf install ffmpeg")
    exit(1)
except subprocess.CalledProcessError:
    print("\n❌ Erro ao converter o vídeo. Verifique se o arquivo de vídeo é válido.")
    exit(1)

# 3. Executa a transcrição
print("\n🚀 Carregando o modelo Whisper Large-V3 (isso pode demorar uns segundos na primeira vez)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

print("🎙️  Iniciando a transcrição (Português)...")
segments, info = model.transcribe(arquivo_audio, beam_size=5, language="pt")

total_duracao = info.duration
print(f"⏱️  Duração total do vídeo: {str(datetime.timedelta(seconds=int(total_duracao)))}")

with tqdm(total=total_duracao, unit="s", desc="Processando áudio") as pbar:
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        ultimo_pos = 0
        for segment in segments:
            inicio = str(datetime.timedelta(seconds=int(segment.start)))
            fim = str(datetime.timedelta(seconds=int(segment.end)))
            
            linha = f"[{inicio} -> {fim}] {segment.text}\n"
            f.write(linha)
            
            pbar.update(segment.end - ultimo_pos)
            ultimo_pos = segment.end

print(f"\n✅ Transcrição concluída com sucesso!")
print(f"📄 Arquivo salvo em: {arquivo_saida}")

# Limpa o arquivo de áudio temporário para não ocupar espaço
if os.path.exists(arquivo_audio):
    os.remove(arquivo_audio)
