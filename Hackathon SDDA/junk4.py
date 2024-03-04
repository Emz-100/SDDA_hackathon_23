import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib

# Function to train and save both the model and TF-IDF vectorizer
def train_and_save_model(input_text, file_path, model_save_path, vectorizer_save_path):
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

# Function to make a category prediction based on user input
def predict_category(input_text, model_path, vectorizer_path):
    # Load the trained model and TF-IDF vectorizer
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)

    # Vectorize the input text using the loaded TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Make predictions
    prediction = model.predict(input_tfidf)
    confidence = np.max(model.predict_proba(input_tfidf))
    confidence_threshold = 0.06666666666666667

    if confidence > confidence_threshold:
        return prediction[0], confidence
    return None, confidence_threshold

def main():
    user_input = "tomatoes, maize meal, curry powder, boerewors"
    csv_file = "Recipes.xlsx"
    model_save_path = 'recipe_model.joblib'
    vectorizer_save_path = 'tfidf_vectorizer.joblib'

    # Train and save the model and TF-IDF vectorizer
    train_and_save_model(user_input, csv_file, model_save_path, vectorizer_save_path)

    # Make predictions using the saved model and vectorizer
    recipe_predicted, confidence = predict_category(user_input, model_save_path, vectorizer_save_path)
    
    print(f"Predicted Recipe: {recipe_predicted} with a confidence of {confidence}")
    print("=" * 20)

if __name__ == "__main__":
    main()
