import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import numpy as np
import av
import io
import wave
import time
import queue
import speech_recognition as sr
from gtts import gTTS
import base64
import random
import streamlit.components.v1 as components

# Configuración de ICE (STUN y TURN)
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:relay1.expressturn.com:3478"],
            "username": "efBBNFJF1809NYOOA7",
            "credential": "mFYpag3yUL3bHv9j"
        }
    ]
}

# Título y descripción
st.title("Conversación de Voz a Texto y Síntesis de Voz")
st.markdown("""
Esta aplicación captura tu voz, la convierte a texto y genera una respuesta automáticamente.  
El sistema mantiene el historial de la conversación y, una vez que se responde, vuelve a escuchar.
""")

# Inicializar variables de sesión para el modo, el texto reconocido y el flag mute
if "mode" not in st.session_state:
    st.session_state.mode = "idle"  # "idle", "listening" o "responding"
if "texto_dicho" not in st.session_state:
    st.session_state.texto_dicho = ""  # Guardará el texto reconocido
if "mute" not in st.session_state:
    st.session_state.mute = False     # Bloquea la captura de audio mientras se reproduce la respuesta

# --- Definición de AudioProcessor ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def recv_queued(self) -> av.AudioFrame:
        frames = []
        while not self.queued_frames.empty():
            frame = await self.queued_frames.get()
            frames.append(frame)
        if frames:
            self.frames.extend(frames)
            # Retornamos el último frame para mantener la actualización interna
            return frames[-1]
        return None

# --- Inicializar streamlit-webrtc (modo SENDONLY para capturar sin reproducir localmente) ---
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    audio_receiver_size=1024,
)

# --- Función: convertir frames a WAV ---
def frames_to_wav(frames):
    try:
        all_samples = np.concatenate([frame.to_ndarray() for frame in frames], axis=1)
    except Exception as e:
        st.error(f"Error al concatenar frames: {e}")
        return None

    sample_rate = frames[0].sample_rate if frames else 48000
    try:
        n_channels = frames[0].layout.channels
        if not n_channels or n_channels < 1:
            raise ValueError("Número de canales no válido")
    except Exception as e:
        st.warning("No se pudo obtener el número de canales, usando 2 por defecto.")
        n_channels = 2

    wav_bytes_io = io.BytesIO()
    try:
        with wave.open(wav_bytes_io, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)  # 16 bits = 2 bytes
            wf.setframerate(sample_rate)
            if all_samples.dtype != np.int16:
                all_samples = all_samples.astype(np.int16)
            wf.writeframes(all_samples.tobytes())
        wav_bytes_io.seek(0)
        return wav_bytes_io
    except Exception as e:
        st.error(f"Error al crear el archivo WAV: {e}")
        return None

# --- Función: convertir audio capturado a texto ---
def speech_to_text():
    # Verificar si webrtc_ctx y su audio_receiver están disponibles
    if webrtc_ctx is None or webrtc_ctx.audio_receiver is None:
        st.error("El componente de audio no está disponible. Asegúrate de que el navegador se haya conectado correctamente.")
        return None

    # Si está en "mute" se ignora la captura
    if st.session_state.mute:
        return None

    # Esperamos unos segundos para acumular audio
    time.sleep(3)
    try:
        frames = webrtc_ctx.audio_receiver.get_frames(timeout=3)
    except queue.Empty:
        st.error("No se recibió audio en el tiempo esperado.")
        return None

    if not frames:
        st.error("No hay frames de audio para procesar.")
        return None

    wav_file = frames_to_wav(frames)
    if not wav_file:
        st.error("Error al crear el archivo de audio.")
        return None

    reconocedor = sr.Recognizer()
    try:
        with sr.AudioFile(wav_file) as source:
            audio_data = reconocedor.record(source)
        texto = reconocedor.recognize_google(audio_data, language="es-ES")
        return texto
    except sr.UnknownValueError:
        st.error("No se entendió lo que dijiste.")
        return None
    except sr.RequestError as e:
        st.error(f"Error en el servicio de reconocimiento: {e}")
        return None


# --- Función: sintetiza respuesta con gTTS y retorna texto y audio en base64 ---
def responder_con_gTTS(texto):
    if "hola" in texto.lower():
        resp = "¡Hola! ¿Cómo puedo ayudarte?"
    elif "adiós" in texto.lower():
        resp = "Adiós, que tengas un buen día."
    else:
        resp = "No estoy seguro de cómo responder a eso."
    try:
        tts = gTTS(resp, lang="es")
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
        return resp, audio_base64
    except Exception as e:
        st.error(f"Error al sintetizar la voz: {e}")
        return resp, None

# --- Función: inyecta HTML para reproducir audio automáticamente con un ID único ---
def reproducir_audio_autoplay(audio_base64):
    unique_id = random.randint(0, 1000000)
    html_code = f"""
        <audio id="audio_{unique_id}" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Tu navegador no soporta la reproducción automática.
        </audio>
        <script>
            var audioElem = document.getElementById("audio_{unique_id}");
            if (audioElem) {{
                audioElem.play();
            }}
        </script>
    """
    components.html(html_code, height=100)

# --- Modo conversacional ---
# Modo "idle": mostramos un botón para iniciar la conversación.
if st.session_state.mode == "idle":
    if st.button("Iniciar Conversación"):
        st.session_state.mode = "listening"
        st.experimental_rerun()

# Modo "listening": se graba y convierte el audio a texto.
if st.session_state.mode == "listening":
    # Si se está reproduciendo audio (mute activo), se espera para evitar feedback.
    if st.session_state.mute:
        st.info("Esperando a que termine la reproducción...")
        time.sleep(5)
        st.experimental_rerun()
    else:
        st.info("Escuchando... Por favor, habla ahora.")
        texto_dicho = speech_to_text()
        if texto_dicho is not None:
            st.session_state.texto_dicho = texto_dicho
            st.session_state.mode = "responding"
        st.experimental_rerun()

# Modo "responding": se genera la respuesta y se reproduce.
if st.session_state.mode == "responding":
    st.info("Generando respuesta...")
    # Activamos el mute para evitar que la respuesta se capture
    st.session_state.mute = True
    respuesta_texto, audio_base64 = responder_con_gTTS(st.session_state.texto_dicho)
    st.write(f"**Usuario:** {st.session_state.texto_dicho}")
    st.write(f"**Respuesta:** {respuesta_texto}")
    if audio_base64:
        reproducir_audio_autoplay(audio_base64)
    # Se espera un tiempo suficiente para que se reproduzca el audio
    time.sleep(3)
    # Desactivamos el mute y volvemos al modo de escucha
    st.session_state.mute = False
    st.session_state.mode = "listening"
    st.experimental_rerun()







