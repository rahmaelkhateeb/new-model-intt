import numpy as np
import pickle
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv

try:
    import tensorflow as tf
    from audio_processor import AudioEmotionProcessor
    AUDIO_SUPPORT = True
except ImportError:
    print("Warning: TensorFlow or librosa not installed. Audio emotion detection disabled.")
    AUDIO_SUPPORT = False

load_dotenv()

class EmotionDetector:
    def __init__(self, gemini_api_key):
        if AUDIO_SUPPORT:
            try:
                self.model = tf.keras.models.load_model('models/emotion_model.h5')
                with open('models/processor.pkl', 'rb') as f:
                    self.processor = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load audio model: {e}")
                self.model = None
                self.processor = None
        else:
            self.model = None
            self.processor = None
        
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.emotion_intensities = {
            'neutral': 0.3, 'calm': 0.4, 'happy': 0.7,
            'sad': 0.8, 'angry': 0.9, 'fearful': 1.0,
            'disgust': 0.8, 'surprised': 0.5
        }
    
    def detect_text_emotion(self, text):
        prompt = f"""
        Analyze emotion in: "{text}"
        Return ONLY JSON: {{"primary_emotion": "emotion", "intensity": 0.5, "confidence": 0.8}}
        Emotions: neutral,calm,happy,sad,angry,fearful,disgust,surprised
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            return self._parse_gemini_emotion(response.text)
        except Exception as e:
            print(f"Gemini API Error (detect_text_emotion): {e}", flush=True)
            return {'primary_emotion': 'neutral', 'intensity': 0.3, 'confidence': 0.5}
    
    def detect_audio_emotion(self, audio_path):
        features = self.processor.extract_features(audio_path)
        if features is None:
            return {'primary_emotion': 'neutral', 'intensity': 0.3, 'confidence': 0.0}
        
        pred = self.model.predict(features.reshape(1, -1), verbose=0)
        emotion_idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        
        emotion = self.processor.emotions[emotion_idx]
        return {
            'primary_emotion': emotion,
            'intensity': self.emotion_intensities.get(emotion, 0.5),
            'confidence': confidence
        }
    
    def fuse_emotions(self, text_result, audio_result):
        if audio_result['confidence'] > 0.7:
            w_audio, w_text = 0.7, 0.3
        else:
            w_audio, w_text = 0.3, 0.7
        
        return {
            'primary_emotion': audio_result['primary_emotion'],
            'intensity': min(w_audio * audio_result['intensity'] + w_text * text_result['intensity'], 1.0),
            'confidence': (text_result['confidence'] + audio_result['confidence']) / 2
        }
    
    def generate_response(self, user_input, emotion_result):
        emotion = emotion_result['primary_emotion']
        intensity = emotion_result['intensity']
        
        templates = {
            'happy': "Celebrate their joy, encourage positivity",
            'sad': f"Empathize deeply (intensity {intensity:.1f}), validate feelings, gentle hope",
            'angry': f"De-escalate (intensity {intensity:.1f}), validate without agreeing",
            'fearful': f"Reassure safety (intensity {intensity:.1f}), practical steps",
            'neutral': "Engage supportively, build connection"
        }
        
        prompt = f"""Compassionate mental health assistant. {templates.get(emotion, 'Supportive listener')}

User: "{user_input}"

Rules:
1. Acknowledge emotion first
2. 1-2 practical coping strategies  
3. Positive, hopeful ending
4. < 120 words
5. Natural, human tone

Response:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error (generate_response): {e}", flush=True)
            return f"I hear you're feeling {emotion}. That's valid. Try deep breathing: inhale 4s, hold 4s, exhale 4s. You're stronger than you know 💪"
    
    def _parse_gemini_emotion(self, text):
        try:
            json_match = re.search(r'\{[^}]*\}', text)
            if json_match:
                import json
                result = json.loads(json_match.group())
                return {
                    'primary_emotion': result.get('primary_emotion', 'neutral'),
                    'intensity': float(result.get('intensity', 0.5)),
                    'confidence': float(result.get('confidence', 0.5))
                }
        except:
            pass
        return {'primary_emotion': 'neutral', 'intensity': 0.3, 'confidence': 0.5}