import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib

# Function to train and save both the model and TF-IDF vectorizer
def train_and_save_model(file_path, model_save_path, vectorizer_save_path):
    # Create a DataFrame from the sample data
    df = pd.read_excel(file_path)

    # Vectorize the text data using TF-IDF (Term Frequency-Inverse Document Frequency)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['Ingredients'])
    y_recipe = df['Recipe']

    # Train a Multinomial Naive Bayes classifier for category prediction
    model = MultinomialNB()
    model.fit(X_tfidf, y_recipe)

    # Save the trained model and vectorizer
    joblib.dump(model, model_save_path)
    joblib.dump(tfidf_vectorizer, vectorizer_save_path)

# Function to make N recipe predictions based on user input
def predict_category(input_text, model_path, vectorizer_path, N=3):
    # Load the trained model and TF-IDF vectorizer
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)

    # Vectorize the input text using the loaded TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Get probabilities for each recipe
    probabilities = model.predict_proba(input_tfidf)[0]

    # Get the top N categories with the highest probabilities
    top_indices = np.argsort(-probabilities)[:N]  # Sort in descending order and get top N indices
    top_categories = [model.classes_[i] for i in top_indices]  # Get recipe labels
    top_probabilities = probabilities[top_indices]  # Get corresponding probabilities

    return top_categories

   


