# src/predict.py
import pickle  
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class SpamClassifier:
    def __init__(self):
        with open('models/spam_classifier_rf.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_email(self, text):
        # Step-by-step processing (no single regex)
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Basic URL removal
        text = re.sub(r'<.*?>', '', text)  # Simple HTML tag removal
        
        # Manual punctuation removal
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Digit removal
        text = ''.join([char for char in text if not char.isdigit()])
        
        # Stopword removal
        tokens = [word for word in text.split() 
                 if word not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def predict(self, email_text):
        cleaned_text = self.clean_email(email_text)
        email_vec = self.vectorizer.transform([cleaned_text])
        pred = self.model.predict(email_vec)[0]
        proba = self.model.predict_proba(email_vec)[0][1]
        
        return {
            'prediction': 'spam' if pred == 1 else 'ham',
            'probability': float(proba),
            'clean_text': cleaned_text
        }

if __name__ == "__main__":
    print("=== Email Classifier (2022) ===")
    test_emails = [
        "Win a free iPhone now! Click here!",
        "Meeting reminder: Project review at 3pm"
    ]
    
    classifier = SpamClassifier()
    for email in test_emails:
        result = classifier.predict(email)
        print(f"\nEmail: {email[:50]}...")
        print(f"Cleaned: {result['clean_text'][:50]}...")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['probability']:.1%}")