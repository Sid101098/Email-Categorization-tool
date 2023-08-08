
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os 
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_email(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_email)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Processed {len(df)} emails")

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_dir, 'data', 'spam-email-dataset', 'emails.csv')
    output_path = os.path.join(project_dir, 'data', 'spam-email-dataset', 'cleaned_emails.csv')
    
    preprocess_data(input_path, output_path)