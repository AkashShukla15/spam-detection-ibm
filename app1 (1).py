import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import os

# Download necessary NLTK data safely
nltk.download('stopwords', quiet=True)

# Set up preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize, remove stopwords, and apply stemming
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Load the saved model and vectorizer
model_path = os.path.join('models', 'spam_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error("Saved model or vectorizer not found. Please run 'train1 (1).py' first.")
    st.stop()

# Streamlit Interface UI Layout
st.title("Spam Detection using Naive Bayes")
user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Process the input using the identical training pipeline
        cleaned_msg = preprocess_text(user_input)
        
        # IMPORTANT: Use transform(), never fit_transform() here
        vectorized_msg = vectorizer.transform([cleaned_msg])
        
        # Run prediction index
        prediction = model.predict(vectorized_msg)[0]
        
        # Display the accurate outcome flag
        if prediction == 'spam' or prediction == 1:
            st.error("🚨 Spam Detected!")
        else:
            st.success("✅ Ham (Not Spam)")
    else:
        st.warning("Please enter a message to evaluate.")
