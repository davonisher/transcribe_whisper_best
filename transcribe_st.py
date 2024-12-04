import streamlit as st
import tempfile
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from groq import Groq

# Setting the Streamlit app title and page configuration
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎧",
    layout="centered"
)

# Sidebar with information about the app
with st.sidebar:
    st.title("About this App")
    st.info("""
    **Audio Transcription App**

    - This app transcribes uploaded audio files to text.
    - Supports multiple audio formats: WAV, MP3, M4A, OGG, FLAC.
    """)

# Loading the configuration file for authentication
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initializing the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Creating the login widget
try:
    authenticator.login('main')
except Exception as e:
    st.error(e)

# Checking the authentication status
if st.session_state['authentication_status']:
    # Add a logout button
    authenticator.logout('Logout')

    # Main title
    st.title(f"🎤 Welcome, {st.session_state['name']}!")

    # OpenAI API key (ensure you store this securely)
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def get_groq_client():
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    # Initialize the Groq client
    groq_client = get_groq_client()

    # File uploader with an enhanced interface
    st.subheader("Upload your audio file")
    uploaded_file = st.file_uploader(
        "Choose an audio file to transcribe...",
        type=["wav", "mp3", "m4a", "ogg", "flac"]
    )

    if uploaded_file is not None:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Display a waiting indicator while transcribing
        with st.spinner('Transcribing...'):
            # Transcribe the audio
            try:
                with open(temp_file_path, "rb") as file:
                    translation = groq_client.audio.translations.create(
                        file=(uploaded_file.name, file.read()),
                        model="whisper-large-v3",
                        prompt="",
                        response_format="json",
                        temperature=0.0
                    )
                # Display the transcription
                st.success("Transcription completed!")
                st.subheader("Transcription:")
                st.write(translation.text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(temp_file_path)
    else:
        st.info("Upload an audio file to begin.")

elif st.session_state['authentication_status'] is False:
    st.error('Username or password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
