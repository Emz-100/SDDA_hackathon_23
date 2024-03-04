import recipe_model_and_trainer 

def main():
    user_input = "apple banana"
    model_save_path = 'recipe_model.joblib'
    vectorizer_save_path = 'tfidf_vectorizer.joblib'

    csv_file = "Recipes.xlsx"
    model_save_path = 'recipe_model.joblib'
    vectorizer_save_path = 'tfidf_vectorizer.joblib'

    # Train and save the model and TF-IDF vectorizer
    #recipe_model.train_and_save_model( csv_file, model_save_path,vectorizer_save_path )
    print("Model trained and Saved")



    # Make predictions using the saved model and vectorizer
    recipe_predicted = recipe_model_and_trainer.predict_category(user_input, model_save_path, vectorizer_save_path,N=3)
    
    print(f"Predicted Recipe: {recipe_predicted} ")
    print("=" * 20)

if __name__ == "__main__":
    main()
