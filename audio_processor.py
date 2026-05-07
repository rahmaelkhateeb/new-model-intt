import librosa
import numpy as np
import os

class AudioEmotionProcessor:
    def __init__(self, sr=22050, n_mfcc=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 
                        'fearful', 'disgust', 'surprised']
        
    def extract_features(self, audio_path, max_length=174):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=4.0)
            
            # MFCC (40)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Additional features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Pad/truncate
            mfccs_padded = np.pad(mfccs, ((0, max(0, max_length - mfccs.shape[1])), (0,0)), 
                                mode='constant')[:, :max_length]
            
            # Statistics
            features = np.hstack([
                np.mean(mfccs_padded, axis=1), np.std(mfccs_padded, axis=1),
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
                np.mean(chroma, axis=1),
                np.mean(spectral_contrast, axis=1)
            ])
            
            return features
            
        except Exception as e:
            print(f"Audio error: {e} - audio_processor.py:43")
            return None
    
    def prepare_ravdess_dataset(self, ravdess_path):
        X, y = [], []
        for root, _, files in os.walk(ravdess_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    emotion = self._get_ravdess_emotion(file)
                    
                    features = self.extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
        return np.array(X), np.array(y)
    
    def _get_ravdess_emotion(self, filename):
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        code = filename.split('-')[2]
        return emotion_map.get(code[:2], 'neutral')