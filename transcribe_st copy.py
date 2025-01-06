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

    - This app transcribes uploaded audio files to text using the current fastest transcription methods and state-of-the-art large language models.
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
    st.title(f"🎤 Welcome, {st.session_state['name']}!")
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
                
                # Generate a summary and to-do list using LLM 3.2
                llama_client = Groq()
                chat_completion = llama_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Summarize the following text and generate a to-do list: {translation.text}"}
                    ],
                    model="llama-3.2-90b-vision-preview",
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                    stop=None,
                    stream=False,
                )

                # Extract and display the summary and to-do list
                response_content = chat_completion.choices[0].message.content
                st.subheader("Summary and To-Do List:")
                st.write(response_content)

                # Display the transcription
                st.success("Transcription completed!")
                st.subheader("Transcription:")
                st.write(translation.text)

                # Save the transcription and summary to files
                with open("transcription.txt", "w") as trans_file:
                    trans_file.write(translation.text)
                with open("summary_and_todo.txt", "w") as summary_file:
                    summary_file.write(response_content)

                st.info("Transcription and summary have been saved to files.")

                # Add download buttons for the transcription and summary
                st.download_button(
                    label="Download Transcription",
                    data=translation.text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                st.download_button(
                    label="Download Summary and To-Do List",
                    data=response_content,
                    file_name="summary_and_todo.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(temp_file_path)
    else:
        st.info("Upload an audio file to begin.")

    # Add a logout button
    authenticator.logout('Logout')

elif st.session_state['authentication_status'] is False:
    st.error('Username or password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
