import os
import spacy
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ✅ Ensure spaCy model is installed
os.system("python -m spacy download en_core_web_sm")

# ✅ Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load intents from the JSON file
with open('law_intents.json') as file:
    intents = json.load(file)

# Initialize patterns, responses, and tags
patterns = []
responses = []
tags = []

# Process the intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])  # Use 'responses' instead of 'response'
        tags.append(intent['tag'])

# Function to preprocess text: Tokenization and Lemmatization using spaCy
def preprocess_text(text):
    doc = nlp(text)  # Process text with spaCy
    return [token.lemma_.lower() for token in doc if token.is_alpha]  # Lemmatize and remove non-alphabetic tokens

# Create the feature vectorizer using TF-IDF
vectorizer = TfidfVectorizer(analyzer=preprocess_text)
X = vectorizer.fit_transform(patterns)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, tags, test_size=0.2, random_state=42)

# Train a Support Vector Classifier (SVC) for text classification
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Function to predict the tag from user input
def predict_tag(user_input):
    input_vector = vectorizer.transform([user_input])  # Convert input into vector form
    return classifier.predict(input_vector)[0]  # Return predicted tag

# Function to get the response based on the predicted tag
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])  # Use 'responses' instead of 'response'

# Streamlit interface
st.title("🧑‍⚖️ Civil Law Chatbot")
st.write("Ask me anything about civil laws, rights, and legal procedures!")

# User input
user_input = st.text_input("You: ", "")

# If user submits a message
if user_input:
    tag = predict_tag(user_input)  # Predict the tag
    response = get_response(tag)   # Get the response
    st.write("Bot: " + response)   # Display the response
