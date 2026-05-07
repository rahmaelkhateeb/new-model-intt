import streamlit as st
import os
import tempfile
from emotion_detector import EmotionDetector

# Page configuration
st.set_page_config(page_title="AI Audio Therapist", page_icon="🧘", layout="centered")

# Custom CSS for a soothing UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e0eaf5 100%);
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .response-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
        margin-top: 20px;
        font-size: 1.1em;
        line-height: 1.6;
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🧘 AI Audio Therapist</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; margin-bottom: 40px;'>Upload a short voice recording (.wav) to receive a therapeutic response tailored to your mood.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return EmotionDetector()

detector = load_detector()

# File uploader
uploaded_file = st.file_uploader("Drop your audio here", type=['wav'])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Action button
    if st.button("Analyze Emotion & Consult Therapist", use_container_width=True):
        
        # Check if model exists
        if detector.model is None:
            st.error("Model not found! Have you trained the model by running `python train_model.py` first?")
        else:
            with st.spinner("Analyzing your voice for emotional cues..."):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    
                # Predict emotion
                emotion = detector.predict_emotion(tmp_path)
                
                # Clean up temp file
                os.remove(tmp_path)
            
            st.success(f"Detected Emotional Tone: **{emotion.capitalize()}**")
            
            with st.spinner("Therapist is typing..."):
                response = detector.generate_therapeutic_response(emotion)
                
                st.markdown("### 💬 Therapist Response")
                st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
