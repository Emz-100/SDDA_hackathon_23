import pandas as pd
import os
import json
from PIL import Image

def find_recipe_info(recipe_name):
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel('Recipes.xlsx')

       # Search for the recipe name in the 'Recipe' column
    recipe_data = df[df['Recipe'] == recipe_name]

    if recipe_data.empty:
        return None  # Recipe not found

    # Convert the matched row to a dictionary
    recipe_dict = recipe_data.iloc[0].to_dict()

    # Construct the JSON object
    json_result = {
        "Recipe": recipe_dict["Recipe"],
        "Ingredients": recipe_dict["Ingredients"],
        "Instructions": recipe_dict["Instructions"],
        "Cooking time": recipe_dict["Cooking time"],
        "Health rating": recipe_dict["Health rating"],
        "Category": recipe_dict["Category"]
    }

    # Check if an image with the same name exists in the local folder
    image_filename = f"images/Grilled Pap and Chicken.jpg"  # Assuming the images are in JPG format
    if os.path.isfile(image_filename):
        # Open and convert the image to base64 for embedding in JSON
        with open(image_filename, 'rb') as image_file:
            image_data = image_file.read()
            #json_result["Image"] = image_data.hex()
            json_result["Image"] = "images/Grilled Pap and Chicken.jpg" 

    return json_result

# Example usage:
recipe_name = "Maccaroni and Mince"  # Replace with the desired recipe name
recipe_info = find_recipe_info(recipe_name)
print(recipe_info)

