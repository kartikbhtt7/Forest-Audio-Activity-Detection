import streamlit as st
import numpy as np
import tempfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from predict import load_model, predict, class_names
import torch

torch.classes.__path__ = []

# Configure page
st.set_page_config(
    page_title="Forest Sound Guardian",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model with error handling
@st.cache_resource
def load_app_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# App header
st.title("🌳 Forest Sound Guardian")
st.markdown("""
Detect unusual human-related sounds in forested regions to prevent illegal activities.
Supported sounds: {}
""".format(", ".join(class_names)))

# Load model
with st.spinner("Loading model..."):
    model = load_app_model()

# Initialize session state
if 'input_method' not in st.session_state:
    st.session_state.input_method = 'upload'

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    input_method = st.radio(
        "Select Input Method:",
        ("Upload Audio", "Record Audio"),
        index=0 if st.session_state.input_method == 'upload' else 1,
        key='input_method_selector'
    )
    st.session_state.input_method = 'upload' if input_method == "Upload Audio" else 'record'
    
    st.markdown("---")
    st.markdown("**Detection Classes:**")
    
    # Group classes by category
    human_sounds = ['footsteps', 'coughing', 'laughing', 'breathing', 
                   'drinking_sipping', 'snoring', 'sneezing']
    tool_sounds = ['chainsaw', 'hand_saw']
    vehicle_sounds = ['car_horn', 'engine', 'siren']
    other_sounds = ['crackling_fire', 'fireworks']
    
    st.markdown("👤 **Human Sounds:** " + ", ".join([s.capitalize() for s in human_sounds]))
    st.markdown("🔨 **Tool Sounds:** " + ", ".join([s.capitalize() for s in tool_sounds]))
    st.markdown("🚗 **Vehicle Sounds:** " + ", ".join([s.capitalize() for s in vehicle_sounds]))
    st.markdown("💥 **Explosives/Firearms/Firecrackers** " + ", ".join([s.capitalize() for s in other_sounds]))
    
    st.markdown("---")
    st.markdown("**About**")
    st.write("This system helps detect human activities in forest areas using audio analysis.")

# Main content area
def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Waveform plot
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Spectrogram plot
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title('Mel Spectrogram')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return y, sr, duration

def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read() if hasattr(audio_file, 'read') else audio_file)
        audio_path = tmp_file.name
    
    try:
        # Visualize audio
        with st.spinner('Analyzing audio...'):
            y, sr, duration = visualize_audio(audio_path)
            st.caption(f"Audio duration: {duration:.2f} seconds")
        
        # Make prediction
        with st.spinner('Making prediction...'):
            class_name, confidence = predict(audio_path, model)
        
        # Display results
        st.subheader("Detection Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Sound", class_name.replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        
        # Show alerts based on class
        human_sounds = ['footsteps', 'coughing', 'laughing', 'breathing', 
                      'drinking_sipping', 'snoring', 'sneezing']
        tool_sounds = ['chainsaw', 'hand_saw']
        
        if class_name in human_sounds:
            st.warning("""
            ⚠️ **Human Activity Detected!**
            Potential human presence in the monitored area.
            """)
        elif class_name in tool_sounds:
            st.error("""
            🚨 **ALERT: Human Tool Detected!**
            Potential illegal logging or activity detected. Consider immediate verification.
            """)
        elif class_name in ['car_horn', 'engine', 'siren']:
            st.warning("""
            ⚠️ **Vehicle Detected!**
            Vehicle sounds detected in the monitored area.
            """)
        elif class_name == 'fireworks':
            st.error("""
            🚨 **ALERT: Fireworks Detected!**
            Potential fire hazard and disturbance to wildlife. Immediate verification required.
            """)
        elif class_name == 'crackling_fire':
            st.error("""
            🚨 **ALERT: Fire Detected!**
            Potential wildfire detected. Immediate verification required.
            """)
        else:
            st.success("✅ Environmental sound detected - no immediate threat")
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.exception(e)
    finally:
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass

# Main processing logic
if st.session_state.input_method == 'upload':
    st.header("Upload Audio File")
    
    sample_col, upload_col = st.columns(2)
    with sample_col:
        st.info("Upload a WAV, MP3 or OGG file with forest sounds")
        st.markdown("""
        **Tips for best results:**
        - Use audio with minimal background noise
        - Ensure the sound of interest is clear
        - 2-3 second clips work best
        """)
    
    with upload_col:
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg"],
            help="Supported formats: WAV, MP3, OGG"
        )
        
    if audio_file:
        st.success("File uploaded successfully!")
        with st.expander("Audio Preview", expanded=True):
            st.audio(audio_file)
        process_audio(audio_file)

else:  # Record mode
    st.header("Record Live Audio")
    
    st.info("""
    Click the microphone button below to record a sound for analysis.  
    **Note:** Please ensure your browser has permission to access your microphone.  
    When prompted, click "Allow" to enable recording.
    """)
    
    recorded_audio = st.audio_input(
        label="Record a sound",
        key="audio_recorder",
        help="Click to record forest sounds for analysis",
        label_visibility="visible"
    )
    
    if recorded_audio:
        st.success("Audio recorded successfully!")
        with st.expander("Recorded Audio", expanded=True):
            st.audio(recorded_audio)
        process_audio(recorded_audio)
    else:
        st.write("Waiting for recording...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px;">
    <p>Forest Sound Guardian v1.0 | 🌳 Protect Natural Ecosystems</p>
    <p><small>Built with Streamlit and PyTorch</small></p>
</div>
""", unsafe_allow_html=True)