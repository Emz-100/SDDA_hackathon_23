import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np

# Sample recipe data with health labels (1-5)
data = {
    'text': [
        "500 g lean mince, 3 large onions, grated, 2 ripe tomatoes, 500 g cheese, grated, 250 g butter, salt and pepper to taste, 500 ml milk, 5 eggs, parsley to taste, oregano to taste",
        "500 g chicken, cut into chunks, 2 tbsp oil, 1 onion, chopped, 2 garlic cloves, crushed, 1 tsp paprika, 1 tsp curry powder, 1 tsp salt, 1/4 tsp black pepper, 1/4 cup tomato paste, 2 cups chicken stock, 2 cups maize meal, 4 cups water, 2 tbsp butter, Parsley chopped for garnish",
        "500 g mutton, cut into pieces, 100 ml oil, 1 bay leaf, 3 cinnamon sticks, 1 teaspoon of fennel seeds, 1 large onion, chopped or grated, 1 sprig curry leaves, 2 teaspoons of ginger and garlic paste, 2 teaspoons salt, 1/4 teaspoon of turmeric, 4 tablespoons of masala, 3 potatoes, cut into cubes, 3 cups rice soaked for an hour then washed and steamed with a quarter teaspoon of turmeric and salt for 10 minutes, 3 sprigs of fresh coriander, chopped",
        "500 g mutton cut into chunks, oil for frying, onion chopped, garlic minced, curry powder to taste (about 2 tablespoons), water as needed (about 2 cups), salt and pepper to taste, carrots sliced",
        # Add more recipes here
    ],
    'category': ['Maccaroni and Mince', 'Pap and chicken', 'Rice and Mutton', 'Jamaica curry'],
    'health_range': [3, 4, 2, 5]  # Health labels (1-5)
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Vectorize the text data using TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['Ingredients'])
y_category = df['category']
y_health_range = df['health_range']

# Train a Multinomial Naive Bayes classifier for category prediction
clf_category = MultinomialNB()
clf_category.fit(X_tfidf, y_category)

# Function to make a category prediction based on user input
def predict_category(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    prediction = clf_category.predict(input_tfidf)
    confidence = np.max(clf_health_range.predict_proba(input_tfidf))

    if(confidence>0.3333333333333333):
        return prediction[0], confidence
    
    return None, 0.3
    

# Split the dataset into training and testing sets for health range prediction
X_train_hr, X_test_hr, y_train_health_range, y_test_health_range = train_test_split(X_tfidf, y_health_range, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier for health range prediction
clf_health_range = MultinomialNB()
clf_health_range.fit(X_train_hr, y_train_health_range)

# Function to make a health range prediction based on user input
def predict_health_range(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    prediction = clf_health_range.predict(input_tfidf)
    return prediction[0]


# Take user input and provide both category and health range predictions
while True:
    user_input = input("Enter a recipe description (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    category_prediction, confidence = predict_category(user_input)
    health_range_prediction = predict_health_range(user_input)
    
    print(f"Predicted category: {category_prediction} with a confidence of {confidence}")
    print(f"Predicted health range: {health_range_prediction}")
