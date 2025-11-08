# whisper_simple.py
import os
import re
import torch
import whisper
import logging
from datetime import datetime
import argparse

# Configuracion RX 6600
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['GPU_MAX_ALLOC_PERCENT'] = '90'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Configurar logging
def configurar_logging():
    ruta_logs = "~/Videos/transcription_tools"
    if not os.path.exists(ruta_logs):
        os.makedirs(ruta_logs)
    
    archivo_log = os.path.join(ruta_logs, "transcriptions_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(archivo_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def sanitizar_nombre_archivo(nombre):
    """Sanitiza el nombre del archivo removiendo caracteres especiales"""
    # Remover caracteres no permitidos y reemplazar espacios
    nombre_limpio = re.sub(r'[<>:"/\\|?*]', '', nombre)
    nombre_limpio = re.sub(r'\s+', '_', nombre_limpio.strip())
    # Limitar longitud del nombre
    return nombre_limpio[:100] if len(nombre_limpio) > 100 else nombre_limpio

def generar_nombre_unico(ruta_carpeta, nombre_base):
    """Genera un nombre único para evitar duplicados"""
    contador = 1
    nombre_original = nombre_base

    while os.path.exists(ruta_carpeta):
        nombre_base = f"{nombre_original}_{contador}"
        ruta_carpeta = ruta_carpeta.replace(nombre_original, nombre_base)
        contador += 1

    return ruta_carpeta, nombre_base

def formatear_timestamp(segundos):
    """Convierte segundos a formato HH:MM:SS"""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"

def agrupar_por_bloques(segments, intervalo_segundos=30):
    """Agrupa segmentos en bloques de tiempo específico"""
    bloques = []
    bloque_actual = {
        'inicio': 0,
        'fin': 0,
        'texto': []
    }

    for segment in segments:
        # Obtener el texto del segmento
        texto_segmento = segment['text'].strip()
        tiempo_fin_segmento = segment['end']

        # Si el segmento supera el intervalo, crear nuevo bloque
        if tiempo_fin_segmento > bloque_actual['inicio'] + intervalo_segundos:
            # Guardar bloque actual si tiene contenido
            if bloque_actual['texto']:
                bloque_actual['fin'] = segment['start']
                bloques.append(bloque_actual)

            # Iniciar nuevo bloque
            bloque_actual = {
                'inicio': segment['start'],
                'fin': tiempo_fin_segmento,
                'texto': [texto_segmento]
            }
        else:
            # Agregar al bloque actual
            bloque_actual['texto'].append(texto_segmento)
            bloque_actual['fin'] = tiempo_fin_segmento

    # Agregar último bloque
    if bloque_actual['texto']:
        bloques.append(bloque_actual)

    return bloques

def full_transcription_task(filename_audio_or_video, whisper_size='medium', lang='es', with_timestamps=True, verbose=True):
    # Configurar logging
    configurar_logging()
    
    print("🚀 Starting Transcription with OpenAI Whisper + Graphics Card")
    logging.info(f"Starting Transcription from file: {filename_audio_or_video}")
    
    if not os.path.exists(filename_audio_or_video):
        error_msg = f"❌ Error: Cannot found {filename_audio_or_video}"
        print(error_msg)
        logging.error(error_msg)
        return None

    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"✅ GPU detected: {gpu_name}")
        logging.info(f"GPU detected: {gpu_name}")
        device = "cuda"
    else:
        print("⚠️ Using CPU, fallback")
        logging.warning("GPU not available, using CPU fallback")
        device = "cpu"

    logging.info(f"Loading Whisper model, size {whisper_size}...")
    model = whisper.load_model(whisper_size, device=device)
    logging.info("Whisper Successfully loaded")

    logging.info("Init transcription process...")
    result = model.transcribe(
        filename_audio_or_video,
        language=lang, # Idioma de la transcripción
        task="transcribe",
        fp16=True if device == "cuda" else False,  # Usar fp16 en GPU
        word_timestamps=with_timestamps,  # Activar timestamps por palabra
        verbose=verbose
    )
    logging.info("Full Transcription Done")

    # Crear estructura de carpetas organizadas
    ruta_base = "~/Videos/transcription_tools"
    
    # Crear carpeta base si no existe
    if not os.path.exists(ruta_base):
        os.makedirs(ruta_base)
        print(f"Base Folder created: {ruta_base}")
        logging.info(f"Base Folder created: {ruta_base}")
    
    # Extraer y sanitizar nombre del archivo original
    nombre_archivo_crudo = os.path.basename(filename_audio_or_video).rsplit('.', 1)[0]
    nombre_archivo = sanitizar_nombre_archivo(nombre_archivo_crudo)
    logging.info(f"Full filename sanitized: '{nombre_archivo_crudo}' -> '{nombre_archivo}'")
    
    # Crear fecha en formato dia_mes_año
    fecha_actual = datetime.now().strftime("%d_%m_%Y")
    
    # Crear nombre de carpeta específica
    nombre_carpeta_base = f"transcript-{nombre_archivo}-{fecha_actual}"
    ruta_carpeta_base = os.path.join(ruta_base, nombre_carpeta_base)
    
    # Generar nombre único para evitar duplicados
    ruta_carpeta, nombre_carpeta_final = generar_nombre_unico(ruta_carpeta_base, nombre_carpeta_base)
    
    # Crear carpeta específica
    os.makedirs(ruta_carpeta, exist_ok=True)
    print(f"Created Folder: {nombre_carpeta_final}")
    logging.info(f"Created Folder: {ruta_carpeta}")
    
    # Procesar bloques de tiempo
    bloques = agrupar_por_bloques(result['segments'], intervalo_segundos=30)
    logging.info(f"Divided Transcription in {len(bloques)} blocks from ~30 seconds")

    # Guardar archivo TXT simple (original)
    nombre_salida_txt = os.path.join(ruta_carpeta, f"{nombre_archivo}_transcripcion.txt")
    with open(nombre_salida_txt, "w", encoding="utf-8") as f:
        f.write(result['text'].strip())

    print(f"Transcription saves in format TXT on file : {nombre_salida_txt}")
    logging.info(f"Transcription saves in format TXT on file: {nombre_salida_txt}")

    # Guardar archivo MD con timestamps
    nombre_salida_md = os.path.join(ruta_carpeta, f"{nombre_archivo}_transcripcion_timestamps.md")
    with open(nombre_salida_md, "w", encoding="utf-8") as f:
        f.write(f"# Transcripción: {nombre_archivo}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        f.write(f"**Total time:** {formatear_timestamp(result['segments'][-1]['end'])}\n\n")
        f.write("---\n\n")

        for bloque in bloques:
            inicio_fmt = formatear_timestamp(bloque['inicio'])
            fin_fmt = formatear_timestamp(bloque['fin'])
            texto_bloque = ' '.join(bloque['texto'])

            f.write(f"## [{inicio_fmt} - {fin_fmt}]\n\n")
            f.write(f"{texto_bloque}\n\n")

    print(f"Transcription MD with timestamps saved in : {nombre_salida_md}")
    logging.info(f"Transcription MD saved: {nombre_salida_md}")
    logging.info(f"Total generated blocks : {len(bloques)}")
    logging.info(f"Characters written in file: {len(result['text'])} characters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full_file_path', dest='full_file_path', type=str,
        help="Full path to the file to traslate, Whisper can Manage audio files and video files"
    )
    parser.add_argument(
        '--whisper-size', dest='whisper_size', type=str,
        help="Set whisper size param. Full Documentation in: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages")

    parser.add_argument('--lang', dest='lang', type=str, help="Set the language to traslate, usually 'es' or 'en'")
    parser.add_argument('--with_timestamps', dest='with_timestamps', type=bool, help="Enable to show timestamps in the transcription")
    parser.add_argument('--verbose', dest='verbose', type=bool, help="Show all the process of traslating")
    args = parser.parse_args()
    import sys
    if len(sys.argv) == 0:
        print("Use it like this: python3 transcription-gpu.py full_path_filename.mp4 'medium' 'en' True True")
        sys.exit(1)

    full_file_route = sys.argv[1] or full_file_path

    full_transcription_task(full_file_route, args.whisper_size, args.lang, with_timestamps, verbose)