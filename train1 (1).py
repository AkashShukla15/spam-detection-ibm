import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Download necessary NLTK data for stopwords
nltk.download('stopwords', quiet=True)

# 1. Load the dataset
dataset_path = 'SMSSpamCollection'
df = pd.read_csv(dataset_path, sep='\t', header=None, names=['label', 'message'])

# 2. Print dataset details
print("--- First 5 rows ---")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['label'].value_counts())

# 3. Preprocess every message
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and apply Porter Stemming
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

df['processed_message'] = df['message'].apply(preprocess_text)

# 4. Convert labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 5. Convert text into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_message'])
y = df['label']

# 6. Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the model using MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Save trained model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, os.path.join('models', 'spam_model.pkl'))
joblib.dump(vectorizer, os.path.join('models', 'vectorizer.pkl'))
print("\nModel and vectorizer successfully saved in the 'models' directory.")
