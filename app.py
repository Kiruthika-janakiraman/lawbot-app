import os
import spacy
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Ensure spaCy model is installed
os.system("python -m spacy download en_core_web_sm")

# Load the spaCy model for NLP (English)
nlp = spacy.load('en_core_web_sm')

# Load intents from the JSON file
with open('law_intents.json') as file:
    intents = json.load(file)

# Initialize patterns, responses, and tags
patterns = []
responses = []
tags = []

# Process the intents data
for intent in intents.get('intents', []):  # Use .get() to avoid KeyError
    if 'patterns' in intent and 'responses' in intent and 'tag' in intent:  # Check for missing keys
        for pattern in intent['patterns']:
            patterns.append(pattern)
            responses.append(random.choice(intent['responses']))  # Select one random response
            tags.append(intent['tag'])
    else:
        print(f"Skipping intent due to missing keys: {intent}")

# Check unique tags
unique_tags = set(tags)
print("Unique tags:", unique_tags)
if len(unique_tags) < 2:
    raise ValueError("Dataset must contain at least two unique classes.")

# Function to preprocess text: Tokenization and Lemmatization using spaCy
def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]

# Create the feature vectorizer using TF-IDF
vectorizer = TfidfVectorizer(analyzer=preprocess_text)
X = vectorizer.fit_transform(patterns)

# Train-test split (90-10) without stratification
X_train, X_test, y_train, y_test = train_test_split(X, tags, test_size=0.1, random_state=42)

# Train classifier
if len(unique_tags) > 2:
    classifier = SVC(kernel='linear')
else:
    classifier = KNeighborsClassifier(n_neighbors=3)  # Use KNN for small datasets

classifier.fit(X_train, y_train)

# Function to predict the tag from user input
def predict_tag(user_input):
    input_vector = vectorizer.transform([user_input])
    return classifier.predict(input_vector)[0]

# Function to get the response based on the predicted tag
def get_response(tag):
    for intent in intents.get('intents', []):  # Use .get() to avoid KeyError
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', []))  # Select a random response
    return "Sorry, I don't have an answer for that."

# Streamlit interface
st.title("🧑‍⚖️ Civil Law Chatbot")
st.write("Ask me anything about civil laws, rights, and legal procedures!")

# User input
user_input = st.text_input("You: ", "")

# If user submits a message
if user_input:
    tag = predict_tag(user_input)  # Predict the tag
    response = get_response(tag)   # Get the response
    st.write("Bot: " + response)  # Display the response