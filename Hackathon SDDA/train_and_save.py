import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Function to train and save both the model and TF-IDF vectorizer
def train_and_save_model(file_path, model_save_path,vectorizer_save_path):
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
    joblib.dump(tfidf_vectorizer,vectorizer_save_path)

    # Check if the model file already exists and delete it
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
    # Save the trained model and vectorizer
    joblib.dump(model, model_save_path)
    
def main():

    csv_file = "Recipes.xlsx"
    model_save_path = 'recipe_model.joblib'
    vectorizer_save_path = 'tfidf_vectorizer.joblib'

    # Train and save the model and TF-IDF vectorizer
    train_and_save_model( csv_file, model_save_path,vectorizer_save_path )
    print("Model trained and Saved")

if __name__ == "__main__":
    main()
