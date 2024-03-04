def find_missing_ingredients(ingredient_string, target_list):
    # Split the ingredient string into a list using commas as the delimiter
    ingredients = [ingredient.strip() for ingredient in ingredient_string.split(',')]

    # Initialize an empty list to store missing ingredients
    missing_ingredients = []

    # Iterate through the ingredients and check if each one is in the target list
    for ingredient in ingredients:
        if ingredient not in target_list:
            missing_ingredients.append(ingredient)

    return missing_ingredients

# Your target list of ingredients
target_ingredients = [
    "sugar",
    "cornstarch",
    "apricot nectar",
    "vanilla extract",
    "red apple",
    "bananas",
    "fresh pineapple",
    "fresh strawberries",
    "green grapes"
]

# Your input string
ingredient_string = "1 cup sugar, 1 tablespoon cornstarch, 2 cans (5-1/2 ounces each) apricot nectar, 1 teaspoon vanilla extract, 6 large red apple (coarsely chopped), 8 medium firm bananas (sliced), 1 medium fresh pineapple (peeled and cut into chunks (about 5 cups)), 1 quart fresh strawberries (quartered), 2 cups green grapes"

# Find missing ingredients
missing_ingredients = find_missing_ingredients(ingredient_string, target_ingredients)

# Print the missing ingredients
print("Missing Ingredients:")
for ingredient in missing_ingredients:
    print(ingredient)
