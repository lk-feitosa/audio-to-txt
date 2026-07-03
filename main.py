import datetime
import subprocess
import os
import ctypes
from deep_translator import GoogleTranslator

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

# 2. Solicita o formato e os idiomas
print("\n🌍 Quais legendas você deseja gerar?")
print("1 - Escolher idioma(s) de saída (Manual)")
print("2 - Áudio Misto (Resolve o bug de vídeos que misturam idiomas)")
print("3 - Detectar Automaticamente (Descobre o idioma principal)")
escolha_idioma = input("> ").strip()

idiomas_para_processar = []
duplo_idioma_mode = False
traducao_magica_mode = False

if escolha_idioma == "1":
    print("\nDigite o(s) idioma(s) desejado(s) separados por vírgula (Enter para 'pt').")
    print("Exemplo: pt, en, es")
    langs = input("> ").strip().lower()
    if not langs:
        langs = "pt"
    idiomas_para_processar = [l.strip() for l in langs.split(",") if l.strip()]
    if len(idiomas_para_processar) > 1:
        duplo_idioma_mode = True

elif escolha_idioma == "2":
    traducao_magica_mode = True
    print("\nQuais idiomas você quer salvar no final? (Enter para 'pt, en')")
    langs = input("> ").strip().lower()
    if not langs:
        langs = "pt, en"
    idiomas_para_processar = [l.strip() for l in langs.split(",") if l.strip()]

elif escolha_idioma == "3":
    idiomas_para_processar = [None] # Deixa a IA descobrir o idioma sozinha

else:
    idiomas_para_processar = ["pt"]

# Configurações do arquivo
arquivo_audio = "audio_otimizado.mp3"

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

if traducao_magica_mode:
    print(f"\n✨ [TRADUÇÃO MÁGICA] Extraindo a base perfeita em INGLÊS...")
    
    # 1. Faz a transcrição perfeita em inglês (Impede o Whisper de se confundir em vídeos mistos)
    segments, info = model.transcribe(
        arquivo_audio, 
        beam_size=5, 
        language="en", 
        vad_filter=True,
        condition_on_previous_text=False
    )
    
    total_duracao = info.duration
    print(f"⏱️  Duração total do vídeo: {str(datetime.timedelta(seconds=int(total_duracao)))}")
    
    # Prepara os arquivos de saída e os tradutores do Google
    arquivos_saida = {}
    tradutores = {}
    for lang in idiomas_para_processar:
        nome_arquivo = f"{os.path.splitext(arquivo_video)[0]}_magica_{lang.upper()}.txt"
        arquivos_saida[lang] = open(nome_arquivo, "w", encoding="utf-8")
        if lang != "en":
            tradutores[lang] = GoogleTranslator(source='en', target=lang)
    
    with tqdm(total=total_duracao, unit="s", desc="Extraindo & Traduzindo") as pbar:
        ultimo_pos = 0
        for segment in segments:
            inicio = str(datetime.timedelta(seconds=int(segment.start)))
            fim = str(datetime.timedelta(seconds=int(segment.end)))
            texto_base = segment.text.strip()
            
            # Escreve a mesma linha já traduzida em TODOS os arquivos solicitados simultaneamente!
            for lang in idiomas_para_processar:
                if lang == "en":
                    texto_final = texto_base
                else:
                    try:
                        texto_final = tradutores[lang].translate(texto_base)
                    except Exception:
                        texto_final = texto_base # Falha na tradução (sem internet?), salva original
                
                linha = f"[{inicio} -> {fim}] {texto_final}\n"
                arquivos_saida[lang].write(linha)
                # Garante que o texto está sendo salvo no disco em tempo real
                arquivos_saida[lang].flush() 
            
            pbar.update(segment.end - ultimo_pos)
            ultimo_pos = segment.end
            
    for lang, f in arquivos_saida.items():
        f.close()
        print(f"✅ Arquivo de Tradução Mágica ({lang.upper()}) salvo!")

else:
    # Modo Clássico (Ouvindo o áudio várias vezes através do Whisper)
    for idioma in idiomas_para_processar:
        if duplo_idioma_mode:
            print(f"\n🔄 [MÚLTIPLOS IDIOMAS] Iniciando transcrição para a versão: {idioma.upper()}")
        elif idioma:
            print(f"\n🎙️  Iniciando a transcrição (Idioma forçado: {idioma})...")
        else:
            print("\n🎙️  Iniciando a transcrição (Detectando o idioma principal)...")

        segments, info = model.transcribe(
            arquivo_audio, 
            beam_size=5, 
            language=idioma,
            vad_filter=True,
            condition_on_previous_text=False # Evita que a IA entre em loop e repita a mesma frase várias vezes
        )

        if not idioma:
            print(f"🗣️  Idioma detectado: {info.language} (Probabilidade: {info.language_probability:.2f})")

        total_duracao = info.duration
        if not duplo_idioma_mode or idioma == idiomas_para_processar[0]:
            print(f"⏱️  Duração total do vídeo: {str(datetime.timedelta(seconds=int(total_duracao)))}")

        # Define o nome do arquivo de saída
        if duplo_idioma_mode:
            arquivo_saida = f"{os.path.splitext(arquivo_video)[0]}_transcricao_{idioma.upper()}.txt"
        else:
            arquivo_saida = f"{os.path.splitext(arquivo_video)[0]}_transcricao.txt"

        desc_barra = f"Processando ({idioma.upper()})" if idioma else "Processando áudio"
        
        with tqdm(total=total_duracao, unit="s", desc=desc_barra) as pbar:
            with open(arquivo_saida, "w", encoding="utf-8") as f:
                ultimo_pos = 0
                for segment in segments:
                    inicio = str(datetime.timedelta(seconds=int(segment.start)))
                    fim = str(datetime.timedelta(seconds=int(segment.end)))
                    
                    linha = f"[{inicio} -> {fim}] {segment.text}\n"
                    f.write(linha)
                    
                    pbar.update(segment.end - ultimo_pos)
                    ultimo_pos = segment.end

        print(f"✅ Transcrição concluída! Salvo em: {arquivo_saida}")

print("\n🎉 Processo finalizado com sucesso!")

# Limpa o arquivo de áudio temporário para não ocupar espaço
if os.path.exists(arquivo_audio):
    os.remove(arquivo_audio)
