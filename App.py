# ==============================
# FLASK WEB APP FOR FAKE NEWS DETECTION
# File: app.py
# ==============================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# ==============================
# LOAD TRAINED MODEL
# ==============================

def load_model():
    """Load the trained model from .pkl file"""
    with open("model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    print("⚠️ Model could not be loaded, using fallback logic. Error:", e)
    MODEL_LOADED = False
    model = None

# ==============================
# PREPROCESSING
# ==============================

def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

def predict_fake_news(text):
    processed_text = preprocess_text(text)

    if MODEL_LOADED and model is not None:
        # Use the trained model from .pkl
        prediction_proba = model.predict_proba([processed_text])[0]
        prediction = 'Real' if model.predict([processed_text])[0] == 1 else 'Fake'
        confidence = max(prediction_proba)
    else:
        # Fallback simple keyword + heuristic method
        fake_indicators = ['breaking', 'urgent', 'leaked', 'secret', 'exposed', 'shocking', 'exclusive']
        fake_score = sum([1 for indicator in fake_indicators if indicator in processed_text.lower()])
        if fake_score > 2:
            prediction = 'Fake'
            confidence = 0.65 + (fake_score * 0.1)
        else:
            prediction = 'Real'
            confidence = 0.60 + (len(processed_text.split()) * 0.001)
        confidence = min(confidence, 0.95)

    return prediction, confidence

# ==============================
# ROUTES
# ==============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            text = request.json.get('text', '')
        else:
            text = request.form.get('text', '')
        if not text.strip():
            return jsonify({'error': 'Please provide some text to analyze'}), 400

        prediction, confidence = predict_fake_news(text)
        sentiment, sentiment_score = get_sentiment(text)

        result = {
            'text': text,
            'prediction': prediction,
            'confidence': round(confidence, 3),
            'sentiment': sentiment,
            'sentiment_score': round(sentiment_score, 3),
            'processed_text': preprocess_text(text)
        }
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request'}), 400
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Text field cannot be empty'}), 400

        prediction, confidence = predict_fake_news(text)
        sentiment, sentiment_score = get_sentiment(text)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'message': f'The news appears to be {prediction.lower()} with {confidence:.1%} confidence. Sentiment: {sentiment}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'message': 'Fake News Detection API is running'
    })

# ==============================
# ERROR HANDLERS
# ==============================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==============================
# MAIN
# ==============================

if __name__ == '__main__':
    print("="*50)
    print("🚀 FAKE NEWS DETECTION API")
    print("="*50)
    print("📊 Model Status:", "Loaded ✅" if MODEL_LOADED else "Fallback ⚠️")
    print("🌐 Web Interface: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/api/predict")
    print("💚 Health Check: http://localhost:5000/health")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
