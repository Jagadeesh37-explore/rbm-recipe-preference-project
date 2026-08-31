"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset (recipe_data.csv) simulating a recipe
platform where users choose recipes based on combinations of ingredients and
food characteristics.

The dataset is built from 5 "hidden" user-preference archetypes so that a
Restricted Boltzmann Machine trained later has genuine latent structure to
discover:
    1. Vegetarian / Healthy eaters
    2. Non-vegetarian / High-protein eaters
    3. Spicy-food lovers
    4. Sweet-tooth / Dessert lovers
    5. Comfort-food / High-calorie eaters

Each simulated user is randomly assigned to one (sometimes two) archetypes,
and their recipe choices are generated with noise so the data looks organic
rather than perfectly clean.
"""

import numpy as np
import pandas as pd

# Fixed seed -> reproducible dataset every time this script is run
np.random.seed(42)

NUM_RECORDS = 180          # number of rows (recipe-interaction records)
NUM_USERS = 40              # number of distinct simulated users

# Recipe name pool used purely for readability in the final CSV / report
RECIPE_NAME_POOL = [
    "Paneer Butter Masala", "Vegetable Pulao", "Fruit Salad Bowl",
    "Palak Paneer", "Mixed Veg Curry", "Chicken Curry", "Egg Bhurji",
    "Grilled Chicken", "Chicken Biryani", "Boiled Egg Salad",
    "Spicy Chicken Wings", "Chili Paneer", "Hot & Spicy Noodles",
    "Chocolate Brownie", "Gulab Jamun", "Rasgulla", "Sweet Rice Kheer",
    "Chocolate Milkshake", "Cheese Loaded Pizza", "Butter Rice",
    "Cheesy Pasta", "Cream Cheese Sandwich", "Loaded Cheese Fries",
    "Steamed Vegetables", "Quinoa Salad", "Sprouts Bowl",
    "Grilled Fish", "Lentil Soup", "Multigrain Roti with Veggies",
    "Protein Power Bowl"
]

# ---------------------------------------------------------------------------
# Archetype definitions: each archetype defines the PROBABILITY that a given
# binary feature will be turned on (1) for a recipe chosen by a user of that
# archetype. This is what creates learnable co-occurrence patterns.
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "vegetarian_healthy": {
        "Vegetarian": 0.95, "Chicken": 0.02, "Egg": 0.05, "Cheese": 0.20,
        "Milk": 0.35, "Paneer": 0.55, "Rice": 0.40, "Vegetables": 0.85,
        "Fruits": 0.55, "Sugar": 0.10, "Chocolate": 0.05, "Spices": 0.30,
        "Chili": 0.15, "Protein": 0.45, "Healthy": 0.85, "Sweet": 0.10,
        "Spicy": 0.15,
    },
    "nonveg_protein": {
        "Vegetarian": 0.05, "Chicken": 0.75, "Egg": 0.60, "Cheese": 0.20,
        "Milk": 0.15, "Paneer": 0.05, "Rice": 0.45, "Vegetables": 0.30,
        "Fruits": 0.10, "Sugar": 0.05, "Chocolate": 0.05, "Spices": 0.45,
        "Chili": 0.30, "Protein": 0.85, "Healthy": 0.40, "Sweet": 0.05,
        "Spicy": 0.30,
    },
    "spicy_lover": {
        "Vegetarian": 0.35, "Chicken": 0.45, "Egg": 0.20, "Cheese": 0.25,
        "Milk": 0.10, "Paneer": 0.25, "Rice": 0.35, "Vegetables": 0.35,
        "Fruits": 0.05, "Sugar": 0.05, "Chocolate": 0.02, "Spices": 0.90,
        "Chili": 0.90, "Protein": 0.45, "Healthy": 0.25, "Sweet": 0.05,
        "Spicy": 0.90,
    },
    "sweet_tooth": {
        "Vegetarian": 0.70, "Chicken": 0.05, "Egg": 0.15, "Cheese": 0.15,
        "Milk": 0.55, "Paneer": 0.10, "Rice": 0.30, "Vegetables": 0.10,
        "Fruits": 0.35, "Sugar": 0.90, "Chocolate": 0.75, "Spices": 0.05,
        "Chili": 0.02, "Protein": 0.15, "Healthy": 0.15, "Sweet": 0.90,
        "Spicy": 0.05,
    },
    "comfort_food": {
        "Vegetarian": 0.45, "Chicken": 0.35, "Egg": 0.25, "Cheese": 0.80,
        "Milk": 0.40, "Paneer": 0.30, "Rice": 0.65, "Vegetables": 0.20,
        "Fruits": 0.05, "Sugar": 0.35, "Chocolate": 0.20, "Spices": 0.35,
        "Chili": 0.20, "Protein": 0.35, "Healthy": 0.15, "Sweet": 0.30,
        "Spicy": 0.25,
    },
}

FEATURE_COLUMNS = list(next(iter(ARCHETYPES.values())).keys())
ARCHETYPE_NAMES = list(ARCHETYPES.keys())

# Assign each simulated user a primary archetype (their dominant preference)
user_primary_archetype = {
    user_id: np.random.choice(ARCHETYPE_NAMES)
    for user_id in range(1, NUM_USERS + 1)
}

records = []
for record_index in range(NUM_RECORDS):
    user_id = np.random.randint(1, NUM_USERS + 1)
    archetype = user_primary_archetype[user_id]
    probabilities = ARCHETYPES[archetype]

    # Sample each binary feature according to the archetype's probability,
    # this naturally creates correlated feature combinations.
    feature_values = {
        feature: int(np.random.rand() < prob)
        for feature, prob in probabilities.items()
    }

    # Probability that the user "likes" this recipe: higher if the recipe
    # strongly matches their archetype's signature features.
    like_probability = 0.55 + 0.30 * (
        feature_values["Healthy"] + feature_values["Spicy"]
        + feature_values["Sweet"] + feature_values["Protein"]
    ) / 4
    user_likes_recipe = int(np.random.rand() < min(like_probability, 0.95))

    recipe_name = np.random.choice(RECIPE_NAME_POOL)

    record = {
        "User_ID": user_id,
        "Recipe_ID": record_index + 1,
        "Recipe_Name": recipe_name,
        **feature_values,
        "User_Likes_Recipe": user_likes_recipe,
    }
    records.append(record)

recipe_dataframe = pd.DataFrame(records)

# Reorder columns to match the required schema
ordered_columns = ["User_ID", "Recipe_ID", "Recipe_Name"] + FEATURE_COLUMNS + ["User_Likes_Recipe"]
recipe_dataframe = recipe_dataframe[ordered_columns]

output_path = "dataset/recipe_data.csv"
recipe_dataframe.to_csv(output_path, index=False)

print(f"Dataset generated successfully: {output_path}")
print(f"Shape: {recipe_dataframe.shape}")
print(recipe_dataframe.head())
