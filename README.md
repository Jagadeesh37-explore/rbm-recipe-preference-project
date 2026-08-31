# Learning Hidden Recipe Preferences Using Restricted Boltzmann Machines (RBM)

## 1. Problem Statement
A recipe platform has unlabeled combinations of ingredients and user choices and
wants to learn hidden preference features. This project implements a simple
Restricted Boltzmann Machine (RBM) using sample ingredient and user-choice data,
analyzes the hidden-unit activations, compares representations across different
user preferences, and determines the latent preference patterns captured by the RBM.

## 2. Objective
- Simulate a realistic recipe-platform dataset of ingredients and user choices.
- Train an RBM to learn hidden (latent) representations without any manual labels.
- Interpret what each hidden unit has learned by inspecting its weights.
- Compare hidden representations across known preference groups (vegetarian,
  spicy, sweet, healthy) to validate that the RBM discovered meaningful structure.
- Visualize the results in a way suitable for a college project report.

## 3. Dataset Description
`dataset/recipe_data.csv` contains 180 synthetic recipe-interaction records across
40 simulated users. Each record has:
- **Identifiers:** `User_ID`, `Recipe_ID`, `Recipe_Name`
- **17 binary ingredient/characteristic features:** `Vegetarian`, `Chicken`, `Egg`,
  `Cheese`, `Milk`, `Paneer`, `Rice`, `Vegetables`, `Fruits`, `Sugar`, `Chocolate`,
  `Spices`, `Chili`, `Protein`, `Healthy`, `Sweet`, `Spicy`
- **Target-like column:** `User_Likes_Recipe` (1 = liked, 0 = not liked)

The data is generated from five underlying "archetypes" (vegetarian/healthy,
non-vegetarian/protein, spicy-lover, sweet-tooth, comfort-food) with randomized
noise, so realistic co-occurrence patterns exist for the RBM to discover — see
`src/generate_dataset.py`.

## 4. Technologies Used
| Library | Purpose |
|---|---|
| pandas | Loading and manipulating the dataset |
| numpy | Numerical operations and array handling |
| matplotlib / seaborn | Visualizations for the report |
| scikit-learn | `BernoulliRBM` model, `PCA` for cluster visualization |

## 5. Project Structure
```
rbm_recipe_preference_project/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   └── recipe_data.csv
│
├── src/
│   ├── generate_dataset.py
│   └── rbm_recipe_preferences.py
│
├── notebooks/
│   └── RBM_Analysis.ipynb
│
├── results/
│   ├── dataset_preview.png
│   ├── hidden_unit_heatmap.png
│   ├── user_representation_comparison.png
│   ├── latent_preference_patterns.png
│   ├── preference_clusters.png
│   ├── hidden_unit_activations.csv
│   └── user_group_comparison.csv
│
└── screenshots/
    └── output.png
```

## 6. Installation Steps
```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 7. How to Run the Project
```bash
# (Optional) Regenerate the dataset from scratch
python src/generate_dataset.py

# Run the full RBM analysis pipeline
python src/rbm_recipe_preferences.py
```
All console output (dataset exploration, training logs, hidden-unit
interpretation, group comparisons) will print to the terminal, and every chart
will be saved automatically into the `results/` folder.

To explore step-by-step interactively, open the notebook instead:
```bash
jupyter notebook notebooks/RBM_Analysis.ipynb
```

## 8. Expected Output
- Console printout of dataset exploration and RBM training progress.
- `results/hidden_unit_activations.csv`: every recipe record's hidden-unit values.
- `results/user_group_comparison.csv`: average hidden activation by preference group.
- Five PNG visualizations described below.

## 9. Key Findings
- The RBM, trained without any labels, produces hidden units whose strongest
  weights cluster naturally around **sweet ingredients** (Sugar, Chocolate,
  Sweet), **spicy ingredients** (Chili, Spices, Spicy), and **vegetarian/healthy**
  ingredients (Fruits, Vegetables, Vegetarian).
- Grouping recipes by known characteristics (e.g. `Spicy = 1` vs `Spicy = 0`)
  shows a clear difference in average activation on the hidden units that
  correspond to that pattern — confirming the RBM learned meaningful structure.
- A PCA projection of the hidden representations shows recipes with similar
  ingredient profiles positioned closer together in the latent space.

## 10. Conclusion
This project demonstrates that a Restricted Boltzmann Machine can learn useful
latent representations purely from unlabeled binary co-occurrence data. Without
being told any category names, the model's hidden units aligned with intuitive,
human-interpretable food-preference concepts, showing the value of unsupervised
feature learning for real-world recommendation-style problems.
