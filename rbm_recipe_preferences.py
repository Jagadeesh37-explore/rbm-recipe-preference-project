"""
rbm_recipe_preferences.py
---------------------------------------------------------------------------
PROJECT : Learning Hidden Recipe Preferences Using Restricted Boltzmann
          Machines (RBM)
AUTHOR  : Jagadeesh M

DESCRIPTION
-----------
A recipe platform records which ingredients each recipe contains and
whether a user liked it, but it has NO labels describing *why* a user
liked a recipe (no "vegetarian", "spicy-lover" tag, etc). This script
trains a Restricted Boltzmann Machine (RBM) on the raw binary
ingredient/preference data so that it automatically discovers hidden
("latent") preference patterns -- purely from co-occurrence statistics,
with no manual labelling of the hidden units.

The script performs, in order:
    1. Load & explore the dataset
    2. Preprocess data into a binary feature matrix
    3. Train a BernoulliRBM
    4. Extract hidden-unit activations for every recipe record
    5. Interpret each hidden unit by inspecting its learned weights
    6. Compare hidden representations across user-preference groups
    7. Generate all report-ready visualizations
---------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

DATASET_PATH = "dataset/recipe_data.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# The binary ingredient / characteristic columns that will be fed to the RBM.
# These are the "visible units" of the network.
FEATURE_COLUMNS = [
    "Vegetarian", "Chicken", "Egg", "Cheese", "Milk", "Paneer", "Rice",
    "Vegetables", "Fruits", "Sugar", "Chocolate", "Spices", "Chili",
    "Protein", "Healthy", "Sweet", "Spicy",
]

NUM_HIDDEN_UNITS = 6          # number of latent preference patterns to learn
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# STEP 1 & 2: LOAD AND EXPLORE THE DATASET
# ---------------------------------------------------------------------------
def load_and_explore_dataset(path: str) -> pd.DataFrame:
    """Loads the CSV dataset and prints basic exploratory information."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run src/generate_dataset.py first."
        )

    dataframe = pd.read_csv(path)

    print("=" * 70)
    print("STEP 2: DATASET LOADING & EXPLORATION")
    print("=" * 70)
    print("\nFirst 5 rows:\n", dataframe.head())
    print("\nDataset shape:", dataframe.shape)
    print("\nColumn names:", list(dataframe.columns))
    print("\nBasic info:")
    dataframe.info()
    print("\nMissing values per column:\n", dataframe.isnull().sum())

    return dataframe


# ---------------------------------------------------------------------------
# STEP 3: DATA PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_features(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Extracts the binary feature matrix used to train the RBM.

    WHY BINARY INPUT SUITS BernoulliRBM:
    BernoulliRBM models its visible units as Bernoulli (0/1) random
    variables. Our ingredient/preference columns are already naturally
    binary (present/absent, liked/not liked), so no scaling or encoding
    is required -- the data distribution matches exactly what the model
    assumes, which keeps training stable and the results interpretable.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)

    feature_matrix = dataframe[FEATURE_COLUMNS].values.astype(np.float32)

    # Safety check: confirm the data really is binary (0/1) as expected.
    unique_values = np.unique(feature_matrix)
    print("Unique values found in feature matrix:", unique_values)
    if not set(unique_values).issubset({0.0, 1.0}):
        print("Warning: non-binary values detected, clipping to [0, 1] range.")
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

    print("Feature matrix shape (samples x visible units):", feature_matrix.shape)
    return feature_matrix


# ---------------------------------------------------------------------------
# STEP 4: TRAIN THE RBM
# ---------------------------------------------------------------------------
def train_rbm(feature_matrix: np.ndarray) -> BernoulliRBM:
    """Trains a BernoulliRBM with beginner-friendly, well-commented parameters."""
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING THE RESTRICTED BOLTZMANN MACHINE")
    print("=" * 70)

    rbm_model = BernoulliRBM(
        n_components=NUM_HIDDEN_UNITS,  # number of hidden units to learn
        learning_rate=0.05,             # step size for weight updates
        batch_size=10,                  # mini-batch size during training
        n_iter=200,                     # number of training epochs
        random_state=RANDOM_STATE,      # reproducibility
        verbose=False,
    )
    rbm_model.fit(feature_matrix)

    print("RBM training complete.")
    print("Visible units (input features):", feature_matrix.shape[1])
    print("Hidden units (learned latent patterns):", NUM_HIDDEN_UNITS)

    return rbm_model


# ---------------------------------------------------------------------------
# STEP 5: EXTRACT HIDDEN-UNIT ACTIVATIONS
# ---------------------------------------------------------------------------
def extract_hidden_activations(
    rbm_model: BernoulliRBM, feature_matrix: np.ndarray, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms every recipe record into the hidden (latent) space and
    returns a tidy DataFrame with one column per hidden unit.
    """
    print("\n" + "=" * 70)
    print("STEP 5: HIDDEN-UNIT ACTIVATIONS")
    print("=" * 70)

    hidden_activations = rbm_model.transform(feature_matrix)  # shape (n_samples, n_hidden)
    hidden_columns = [f"Hidden_Unit_{i + 1}" for i in range(NUM_HIDDEN_UNITS)]

    activation_table = pd.DataFrame(hidden_activations, columns=hidden_columns)
    activation_table.insert(0, "Recipe_Name", dataframe["Recipe_Name"].values)
    activation_table.insert(0, "Recipe_ID", dataframe["Recipe_ID"].values)
    activation_table.insert(0, "User_ID", dataframe["User_ID"].values)

    print(activation_table.head(10))

    activation_table.to_csv(os.path.join(RESULTS_DIR, "hidden_unit_activations.csv"), index=False)
    print(f"\nFull activation table saved to {RESULTS_DIR}/hidden_unit_activations.csv")

    return activation_table


# ---------------------------------------------------------------------------
# STEP 6: INTERPRET LATENT PREFERENCE PATTERNS FROM RBM WEIGHTS
# ---------------------------------------------------------------------------
def interpret_hidden_units(rbm_model: BernoulliRBM, top_n: int = 4) -> None:
    """
    Inspects rbm_model.components_ (shape: n_hidden x n_visible) to see
    which visible features each hidden unit has assigned the strongest
    (most positive) weights to. A large positive weight means that when
    that feature is "on", it strongly excites the hidden unit.

    IMPORTANT: We are NOT telling the RBM what each hidden unit means.
    We are only reading its learned weights afterwards, for interpretation.
    """
    print("\n" + "=" * 70)
    print("STEP 6: INTERPRETING LATENT PREFERENCE PATTERNS")
    print("=" * 70)

    weight_matrix = rbm_model.components_  # (n_hidden, n_visible)

    for hidden_index in range(weight_matrix.shape[0]):
        weights_for_unit = weight_matrix[hidden_index]
        top_feature_indices = np.argsort(weights_for_unit)[::-1][:top_n]
        top_features = [
            (FEATURE_COLUMNS[i], round(float(weights_for_unit[i]), 3))
            for i in top_feature_indices
        ]
        feature_names_only = ", ".join(name for name, _ in top_features)

        print(f"\nHidden Unit {hidden_index + 1}:")
        print(f"  Strongest associated features: {top_features}")
        print(f"  Possible interpretation: This unit may represent a "
              f"preference pattern related to [{feature_names_only}]. "
              f"(Interpretation is based on learned weights, not a predefined label.)")


# ---------------------------------------------------------------------------
# STEP 7: COMPARE HIDDEN REPRESENTATIONS ACROSS USER-PREFERENCE GROUPS
# ---------------------------------------------------------------------------
def compare_user_groups(activation_table: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Groups recipe records by a characteristic (e.g. Vegetarian = 1 vs 0)
    and compares the AVERAGE hidden-unit activation for each group. This
    reveals whether the RBM's latent space actually separates known
    preference groups, even though it was never told these labels.
    """
    print("\n" + "=" * 70)
    print("STEP 7: COMPARING USER PREFERENCE GROUPS")
    print("=" * 70)

    hidden_columns = [c for c in activation_table.columns if c.startswith("Hidden_Unit")]
    comparison_frame = activation_table[hidden_columns].copy()

    comparison_results = {}
    group_definitions = {
        "Vegetarian": dataframe["Vegetarian"],
        "Spicy": dataframe["Spicy"],
        "Sweet": dataframe["Sweet"],
        "Healthy": dataframe["Healthy"],
    }

    for characteristic_name, characteristic_series in group_definitions.items():
        group_yes_mean = comparison_frame[characteristic_series == 1].mean()
        group_no_mean = comparison_frame[characteristic_series == 0].mean()
        comparison_results[f"{characteristic_name}=1"] = group_yes_mean
        comparison_results[f"{characteristic_name}=0"] = group_no_mean

        print(f"\n{characteristic_name} = 1 (Yes) average hidden activations:\n", group_yes_mean.round(3).to_dict())
        print(f"{characteristic_name} = 0 (No)  average hidden activations:\n", group_no_mean.round(3).to_dict())

    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "user_group_comparison.csv"))
    return comparison_df


# ---------------------------------------------------------------------------
# STEP 8: VISUALIZATIONS
# ---------------------------------------------------------------------------
def plot_dataset_preview(dataframe: pd.DataFrame) -> None:
    """Saves a table-style image previewing the first rows of the dataset."""
    preview = dataframe.head(10)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    ax.set_title("Recipe Dataset Preview (First 10 Records)", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "dataset_preview.png"), bbox_inches="tight")
    plt.close(fig)


def plot_hidden_unit_heatmap(activation_table: pd.DataFrame) -> None:
    """Heatmap of hidden-unit activations for a sample of recipe records."""
    hidden_columns = [c for c in activation_table.columns if c.startswith("Hidden_Unit")]
    sample = activation_table[hidden_columns].head(30)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(sample, cmap="viridis", cbar_kws={"label": "Activation Strength"}, ax=ax)
    ax.set_title("Hidden Unit Activations (Sample of 30 Recipe Records)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hidden Units")
    ax.set_ylabel("Recipe Record Index")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "hidden_unit_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def plot_user_group_comparison(comparison_df: pd.DataFrame) -> None:
    """Bar chart comparing average hidden activations across preference groups."""
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.T.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Average Hidden-Unit Activation by User Preference Group", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hidden Units")
    ax.set_ylabel("Average Activation")
    ax.legend(title="Preference Group", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "user_representation_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_latent_preference_patterns(rbm_model: BernoulliRBM) -> None:
    """Heatmap of RBM weights: hidden units vs visible (ingredient) features."""
    weight_matrix = rbm_model.components_

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        weight_matrix,
        cmap="coolwarm",
        center=0,
        xticklabels=FEATURE_COLUMNS,
        yticklabels=[f"Hidden {i + 1}" for i in range(weight_matrix.shape[0])],
        cbar_kws={"label": "Weight Strength"},
        ax=ax,
    )
    ax.set_title("Latent Preference Patterns: Hidden Units vs Ingredient Features", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "latent_preference_patterns.png"), bbox_inches="tight")
    plt.close(fig)


def plot_preference_clusters(activation_table: pd.DataFrame, dataframe: pd.DataFrame) -> None:
    """PCA projection of hidden representations, colored by Vegetarian flag."""
    hidden_columns = [c for c in activation_table.columns if c.startswith("Hidden_Unit")]
    hidden_values = activation_table[hidden_columns].values

    pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coordinates = pca_model.fit_transform(hidden_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_coordinates[:, 0],
        pca_coordinates[:, 1],
        c=dataframe["Vegetarian"],
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        linewidth=0.3,
    )
    legend_labels = ["Non-Vegetarian", "Vegetarian"]
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, title="Recipe Type")
    ax.set_title("Preference Clusters (PCA of Hidden Representations)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "preference_clusters.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main() -> None:
    dataframe = load_and_explore_dataset(DATASET_PATH)
    feature_matrix = preprocess_features(dataframe)

    rbm_model = train_rbm(feature_matrix)

    activation_table = extract_hidden_activations(rbm_model, feature_matrix, dataframe)
    interpret_hidden_units(rbm_model)
    comparison_df = compare_user_groups(activation_table, dataframe)

    print("\n" + "=" * 70)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_dataset_preview(dataframe)
    plot_hidden_unit_heatmap(activation_table)
    plot_user_group_comparison(comparison_df)
    plot_latent_preference_patterns(rbm_model)
    plot_preference_clusters(activation_table, dataframe)
    print(f"All visualizations saved inside the '{RESULTS_DIR}/' folder.")

    print("\nProject run complete. Review the 'results/' folder for all outputs.")


if __name__ == "__main__":
    main()
