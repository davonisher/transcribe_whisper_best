import streamlit as st
import tempfile
from groq import Groq
import os

# Set up the Streamlit app title
st.title("Audio Transcription App with Groq Cloud")

# OpenAI API key (ensure to store this securely in practice)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

# Initialize clients
groq_client = get_groq_client()

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Placeholder for displaying the transcription
    transcription_placeholder = st.empty()

    # Function to process the file (without threading)
    def transcribe_audio():
        try:
            with open(temp_file_path, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(uploaded_file.name, file.read()),
                    model="whisper-large-v3",
                    prompt="",
                    response_format="json",
                    temperature=0.0
                )
            # Update the transcription in the main thread
            transcription_placeholder.write("**Transcription:**")
            transcription_placeholder.write(translation.text)
        finally:
            os.remove(temp_file_path)

    # Call the function directly (synchronously)
    transcribe_audio()
