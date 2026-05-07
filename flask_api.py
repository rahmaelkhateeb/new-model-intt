import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Try to load .env, fallback to .env.example if missing
if not os.path.exists('.env') and os.path.exists('.env.example'):
    load_dotenv('.env.example', override=True)
else:
    load_dotenv(override=True)

from emotion_detector import EmotionDetector

app = Flask(__name__)
CORS(app)

# Initialize the detector
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment. Please set it.")

detector = EmotionDetector(api_key)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    try:
        # Detect emotion
        emotion_result = detector.detect_text_emotion(text)
        
        # Generate response
        response_text = detector.generate_response(text, emotion_result)
        
        return jsonify({
            'emotion': emotion_result,
            'response': response_text
        })
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on port 5001 to avoid conflicting with Node.js on 5000
    app.run(host='0.0.0.0', port=5001, debug=True)
