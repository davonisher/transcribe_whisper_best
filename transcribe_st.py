import streamlit as st
import tempfile
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from groq import Groq

# Instellen van de Streamlit app titel en pagina-configuratie
st.set_page_config(
    page_title="Audio transcription",
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

# Laden van het configuratiebestand voor authenticatie
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialiseren van de authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Creëren van de login widget
name, authentication_status, username = authenticator.login('Inloggen', 'main')

if authentication_status:
    # Als de gebruiker succesvol is ingelogd, toon de applicatie

    # Hoofdtitel
    st.title(f"🎤 Welkom, {name}!")

    # OpenAI API-sleutel (zorg ervoor dat je deze veilig opslaat)
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def get_groq_client():
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    # Initialiseer de Groq client
    groq_client = get_groq_client()

    # Bestandsuploader met een verbeterde interface
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

        # Plaats een wachtindicator tijdens het transcriberen
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

    # Voeg een logout knop toe
    authenticator.logout('Uitloggen', 'sidebar')

elif authentication_status == False:
    st.error('Gebruikersnaam of wachtwoord is onjuist')
elif authentication_status == None:
    st.warning('Voer uw gebruikersnaam en wachtwoord in')

