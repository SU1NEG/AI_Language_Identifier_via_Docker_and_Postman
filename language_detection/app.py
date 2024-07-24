import os
import fasttext
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Dil tanıma modeli yükleme
model = fasttext.load_model('lid.176.bin')

@app.route('/detect_language', methods=['POST'])
def detect_language():
    content = request.get_json(silent=True)
    if not content or 'text' not in content:
        return jsonify({'error': 'No text provided'}), 400

    text = content['text']
    # Dil tanıma
    predictions = model.predict(text, k=1)  # En olası dili tahmin et
    labels, probs = predictions

    # `np.asarray` kullanarak uyumluluğu sağla
    probs = np.asarray(probs, dtype=np.float64)

    language = labels[0].replace('__label__', '')
    confidence = round(float(probs[0]) * 100, 2)

    response = {language: confidence}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
