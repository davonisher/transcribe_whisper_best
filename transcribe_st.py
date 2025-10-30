import streamlit as st
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

import tempfile
from pydub import AudioSegment

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

    # Language selection
    language_options = ["English", "Spanish", "French", "German", "Italian", "Dutch", "Portuguese"]
    selected_language = st.selectbox("Select language for transcription and summary", language_options, index=0)
    selected_language_code = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Dutch": "nl",
        "Portuguese": "pt"
       
    }[selected_language]

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
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)

        # Set a threshold in MB
        threshold_mb = 20

        if file_size_mb > threshold_mb:
            st.warning(f"File is larger than {threshold_mb} MB. Splitting into valid chunks...")
            
            # Use Pydub to split the audio into smaller chunks (time-based)
            try:
                # Load the full audio file
                full_audio = AudioSegment.from_file(temp_file_path, format=uploaded_file.name.split('.')[-1])
                
                # Estimate chunk size (in ms) that yields around threshold_mb each
                # This is approximate, since the exact size depends on bitrate.
                # For example, let's chunk by 10 minutes if the file is quite large.
                chunk_length_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
                
                # Create a list to store each chunk's transcription
                transcription_chunks = []

                # Calculate how many chunks we'll have
                total_duration_ms = len(full_audio)
                num_chunks = (total_duration_ms // chunk_length_ms) + 1

                # Iterate through each chunk of audio
                for i in range(num_chunks):
                    start_ms = i * chunk_length_ms
                    end_ms = min((i + 1) * chunk_length_ms, total_duration_ms)
                    
                    # If start_ms >= total duration, break out
                    if start_ms >= total_duration_ms:
                        break
                    
                    # Extract chunk using slicing
                    audio_chunk = full_audio[start_ms:end_ms]
                    
                    # Export chunk as a valid .m4a file
                    chunk_file_path = f"{temp_file_path}_chunk_{i}.m4a"
                    audio_chunk.export(chunk_file_path, format="m4a")
                    
                    # Now read this chunk in binary mode for transcription
                    with open(chunk_file_path, "rb") as chunk_file:
                        with st.spinner(f'Transcribing chunk {i+1}/{num_chunks}...'):
                            try:
                                transcriptions = groq_client.audio.transcriptions.create(
                                    file=(f"{uploaded_file.name}_chunk{i+1}", chunk_file.read()),
                                    model="whisper-large-v3-turbo",
                                    prompt="",
                                    response_format="json",
                                    temperature=0.0,
                                    language=selected_language_code
                                )
                                transcription_chunks.append(transcriptions.text)
                            except Exception as e:
                                st.error(f"An error occurred while transcribing chunk {i+1}: {e}")
                                break
                    
                    # Remove the chunk file once done
                    os.remove(chunk_file_path)

                # Combine the transcription chunks into a single text
                full_transcription = "\n".join(transcription_chunks)

                # Summarize + to-do list
                llama_client = Groq(api_key=os.environ["GROQ_API_KEY"])
                chat_completion = llama_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are a helpful assistant. Summarize the following text and generate a to-do list in {selected_language}:"},
                        {"role": "user", "content": full_transcription}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                    stop=None,
                    stream=False,
                )

                # Extract and display the summary and to-do list
                response_content = chat_completion.choices[0].message.content

                st.subheader(f"Summary and To-Do List ({selected_language}):")
                st.write(response_content)

                # Display the transcription
                st.success("Transcription completed!")
                st.subheader(f"Transcription ({selected_language}):")
                st.write(full_transcription)

                # Save the transcription and summary to files
                with open("transcription.txt", "w", encoding="utf-8") as trans_file:
                    trans_file.write(full_transcription)
                with open("summary_and_todo.txt", "w", encoding="utf-8") as summary_file:
                    summary_file.write(response_content)

                st.info("Transcription and summary have been saved to files.")

            except Exception as e:
                st.error(f"An error occurred while splitting/transcribing the file: {e}")
            finally:
                os.remove(temp_file_path)

        else:
            # File size is within threshold, we can directly transcribe
            with st.spinner('Transcribing...'):
                try:
                    with open(temp_file_path, "rb") as file:
                        transcriptions = groq_client.audio.transcriptions.create(
                            file=(uploaded_file.name, file.read()),
                            model="whisper-large-v3-turbo",
                            prompt="",
                            response_format="json",
                            temperature=0.0,
                            language=selected_language_code
                        )
                    
                    # Summarize + to-do list
                    llama_client = Groq(api_key=os.environ["GROQ_API_KEY"])
                    chat_completion = llama_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. Summarize the following text and generate a to-do list in {selected_language}:"},
                            {"role": "user", "content": transcriptions.text}
                        ],
                        model="llama-3.2-90b-vision-preview",
                        temperature=0.5,
                        max_tokens=2048,
                        top_p=1,
                        stop=None,
                        stream=False,
                    )

                    response_content = chat_completion.choices[0].message.content

                    st.subheader(f"Summary and To-Do List ({selected_language}):")
                    st.write(response_content)

                    st.success("Transcription completed!")
                    st.subheader(f"Transcription ({selected_language}):")
                    st.write(transcriptions.text)

                    # Save the transcription and summary
                    with open("transcription.txt", "w", encoding="utf-8") as trans_file:
                        trans_file.write(transcriptions.text)
                    with open("summary_and_todo.txt", "w", encoding="utf-8") as summary_file:
                        summary_file.write(response_content)

                    st.info("Transcription and summary have been saved to files.")
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
