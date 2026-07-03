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

# 2. Solicita o idioma do vídeo
print("\n🌍 Qual o idioma do vídeo?")
print("1 - Português (Traduz as outras línguas para PT)")
print("2 - Inglês (Traduz as outras línguas para EN)")
print("3 - Automático (Apenas 1 idioma principal)")
print("4 - Modo Misto (Mantém múltiplos idiomas no original, sem traduzir)")
escolha_idioma = input("> ").strip()

idioma_selecionado = "pt"
multilingual_mode = False

if escolha_idioma == "2":
    idioma_selecionado = "en"
elif escolha_idioma == "3":
    idioma_selecionado = None # Deixa a IA descobrir o idioma principal
elif escolha_idioma == "4":
    idioma_selecionado = None
    multilingual_mode = True # Força a IA a identificar o idioma em CADA frase

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
print("\n🚀 Tentando carregar o modelo Whisper Large-V3 na GPU (NVIDIA)...")
try:
    # Tenta rodar com aceleração máxima na GPU
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("✅ Aceleração de GPU (CUDA) ativada com sucesso!")
except Exception as e:
    print("\n⚠️ Placa de vídeo NVIDIA não detectada ou incompatível.")
    print("🐌 Recuando para processamento na CPU (isso pode ser mais lento)...")
    # Fallback seguro para rodar em literalmente qualquer computador
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

if idioma_selecionado:
    print(f"\n🎙️  Iniciando a transcrição (Idioma forçado: {idioma_selecionado})...")
elif multilingual_mode:
    print("\n🎙️  Iniciando a transcrição (Modo Misto - Múltiplos Idiomas Originais)...")
else:
    print("\n🎙️  Iniciando a transcrição (Detectando o idioma principal)...")

if multilingual_mode:
    # O parâmetro multilingual=True faz a IA redetectar o idioma a cada segmento.
    # O condition_on_previous_text=False impede que frases anteriores num idioma "contaminem" o próximo idioma.
    segments, info = model.transcribe(
        arquivo_audio, 
        beam_size=5, 
        language=None, 
        multilingual=True, 
        condition_on_previous_text=False
    )
else:
    segments, info = model.transcribe(arquivo_audio, beam_size=5, language=idioma_selecionado)

if not idioma_selecionado:
    print(f"🗣️  Idioma detectado: {info.language} (Probabilidade: {info.language_probability:.2f})")

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
