# train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle  
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/spam-email-dataset/cleaned_emails.csv')

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english' 
)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  
)


print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)


print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  
)
rf.fit(X_train, y_train)


def evaluate_model(model, model_name):
    print(f"\n{model_name} Evaluation:")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Ham', 'Spam'])
    plt.yticks([0,1], ['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]),
                    ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.show()

evaluate_model(nb, "Naive Bayes")
evaluate_model(rf, "Random Forest")
print("\nSaving models...")
with open('models/spam_classifier_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
    
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training complete! Models saved to /models/")