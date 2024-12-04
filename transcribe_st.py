import streamlit as st
import tempfile
import os
from groq import Groq

# Instellen van de Streamlit app titel en uitleg
st.set_page_config(
    page_title="Audio Transcription App met Groq Cloud",
    page_icon="🎧",
    layout="centered"
)

# Zijbalk met informatie over de app
with st.sidebar:
    st.title("Over deze App")
    st.info("""
    **Audio Transcription App**

    - Deze app transcribeert geüploade audiobestanden naar tekst.
    - Gebruikt Groq Cloud voor snelle en nauwkeurige transcripties.
    - Ondersteunt meerdere audioformaten: WAV, MP3, M4A, OGG, FLAC.
    """)

# Hoofdtitel
st.title("🎤 Audio Transcription App")

# OpenAI API-sleutel (zorg ervoor dat je deze veilig opslaat in de praktijk)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

# Initialiseer de client
groq_client = get_groq_client()

# Bestandsuploader met een mooiere interface
st.subheader("Upload je audiobestand")
uploaded_file = st.file_uploader(
    "Kies een audiobestand om te transcriberen...",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

if uploaded_file is not None:
    # Opslaan van het geüploade bestand
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Plaats een wachtende indicator tijdens het transcriberen
    with st.spinner('Bezig met transcriberen...'):
        # Transcribeer de audio
        try:
            with open(temp_file_path, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(uploaded_file.name, file.read()),
                    model="whisper-large-v3",
                    prompt="",
                    response_format="json",
                    temperature=0.0
                )
            # Toon de transcriptie
            st.success("Transcriptie voltooid!")
            st.subheader("Transcriptie:")
            st.write(translation.text)
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")
        finally:
            os.remove(temp_file_path)
else:
    st.info("Upload een audiobestand om te beginnen.")

