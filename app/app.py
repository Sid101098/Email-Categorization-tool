# app/app.py
from flask import Flask, render_template, request, jsonify
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.predict import SpamClassifier

app = Flask(__name__)
classifier = SpamClassifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.form.get('email_text'):
        return jsonify({'error': 'No email text provided'}), 400
        
    email_text = request.form['email_text']
    result = classifier.predict(email_text)
    response = jsonify(result)
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)