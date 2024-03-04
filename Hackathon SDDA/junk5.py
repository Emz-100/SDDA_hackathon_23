import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Function to make a category prediction based on user input
def train(input_text, file_path):

    # Create a DataFrame from the sample data
    df = pd.read_excel(file_path)

    # Vectorize the text data using TF-IDF (Term Frequency-Inverse Document Frequency)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['Ingredients'])
    y_recipe = df['Recipe']

    # Train a Multinomial Naive Bayes classifier for category prediction
    model = MultinomialNB()
    model.fit(X_tfidf, y_recipe)
    joblib.dump(model, 'recipe_model.joblib')
    input_tfidf = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    print(f"Size of list: {len(prediction)}")
    confidence = np.max(model.predict_proba(input_tfidf))
    confidence_threshold =0.06666666666666667
    joblib.dump()

    if confidence >confidence_threshold:
        return prediction[0], confidence
    return None, confidence_threshold

def main():
    user_input = "tomatoes, maize meal, curry powder, boerewors"
    csv_file ="Recipes.xlsx"
    recipe_predicted, confidence = train(user_input,csv_file)
    
    print(f"Predicted Recipe: {recipe_predicted} with a confidence of {confidence}")
    print("="*20)



if __name__ =="__main__":
    main()
