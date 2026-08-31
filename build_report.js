const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, PageBreak,
  ImageRun, BorderStyle, TableOfContents, VerticalAlign,
} = require("docx");

const PAGE_WIDTH_DXA = 12240; // US Letter
const PAGE_HEIGHT_DXA = 15840;

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ text, heading: level, spacing: { before: 300, after: 150 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function centered(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
  });
}
function image(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}

const titlePage = [
  new Paragraph({ text: "", spacing: { after: 1600 } }),
  centered("LEARNING HIDDEN RECIPE PREFERENCES", { bold: true, size: 40 }),
  centered("USING RESTRICTED BOLTZMANN MACHINES (RBM)", { bold: true, size: 40 }),
  new Paragraph({ text: "", spacing: { after: 1200 } }),
  centered("A Project Report Submitted in Partial Fulfillment of the", { size: 24 }),
  centered("Requirements for the Degree of", { size: 24 }),
  centered("Bachelor of Engineering in Computer Science and Engineering (AI & ML)", { size: 24, bold: true }),
  new Paragraph({ text: "", spacing: { after: 1000 } }),
  centered("Submitted by", { size: 24 }),
  centered("JAGADEESH M", { size: 28, bold: true }),
  centered("Register Number: [Enter Register Number Here]", { size: 24 }),
  new Paragraph({ text: "", spacing: { after: 1000 } }),
  centered("Department of Computer Science and Engineering (AI & ML)", { size: 24 }),
  centered("[Enter College Name Here]", { size: 26, bold: true }),
  centered("Academic Year: [Enter Academic Year Here]", { size: 24 }),
  new Paragraph({ children: [new PageBreak()] }),
];

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: PAGE_WIDTH_DXA, height: PAGE_HEIGHT_DXA } },
      },
      children: [
        ...titlePage,

        heading("2. Problem Statement"),
        body("A recipe platform collects information about the ingredients used in each recipe and whether a user liked that recipe, but this data has no labels describing why a user liked it -- there is no tag such as \"vegetarian lover\" or \"spicy food fan\" attached to any user. The platform wants to automatically discover these hidden preference patterns directly from the raw, unlabeled ingredient and choice data. This project implements a simple Restricted Boltzmann Machine (RBM) to learn such hidden features, analyzes the resulting hidden-unit activations, compares them across different user preference groups, and interprets the latent patterns that the model has captured."),

        heading("3. Objective"),
        bullet("To design and generate a realistic synthetic dataset simulating a recipe platform's ingredient and preference data."),
        bullet("To implement a Restricted Boltzmann Machine using scikit-learn's BernoulliRBM on this unlabeled binary data."),
        bullet("To extract and study the hidden-unit activations produced by the trained RBM for each recipe record."),
        bullet("To interpret the latent preference pattern represented by each hidden unit using its learned weights."),
        bullet("To compare hidden representations across known preference groups (vegetarian, spicy, sweet, healthy) and validate that the RBM has captured meaningful structure."),
        bullet("To visualize all results in a manner suitable for a college project report."),

        heading("4. Concept Used"),
        body("Unsupervised Learning:", { bold: true }),
        body("Unsupervised learning is a branch of machine learning where the model is trained on data that has no predefined output labels. Instead of predicting a known answer, the model tries to discover hidden structure, patterns, or groupings within the data on its own. This project is unsupervised because the RBM is never told which user prefers which type of food -- it only sees raw ingredient combinations."),
        body("Restricted Boltzmann Machine (RBM):", { bold: true }),
        body("A Restricted Boltzmann Machine is a type of stochastic neural network made up of two layers of neurons: a visible layer and a hidden layer, with connections only between the two layers (no connections within the same layer, hence \"restricted\"). An RBM learns to reconstruct its input data by adjusting the weights between visible and hidden units, and in doing so it learns compressed, meaningful representations of the data."),
        body("Visible Layer:", { bold: true }),
        body("The visible layer represents the raw input features fed into the network. In this project, the visible units correspond to the 17 binary ingredient and characteristic columns such as Vegetarian, Chicken, Spicy, and Sweet."),
        body("Hidden Layer:", { bold: true }),
        body("The hidden layer represents the latent (unobserved) features that the RBM learns during training. Each hidden unit becomes activated based on combinations of visible features, and over time these units begin to specialize in representing particular co-occurring patterns in the data."),
        body("Feature Learning:", { bold: true }),
        body("Feature learning refers to a model's ability to automatically discover useful representations of raw data, instead of relying on manually engineered features. The RBM performs feature learning by converting raw ingredient combinations into a small set of hidden-unit activations that summarize the essential preference pattern of each recipe record."),
        body("Latent Representations:", { bold: true }),
        body("A latent representation is a compressed, hidden encoding of data that captures its most important underlying characteristics. In this project, the hidden-unit activation vector for each recipe record is its latent representation -- a compact numerical summary of its preference pattern."),

        heading("5. Methodology / Working Steps"),
        bullet("Step 1: Generate a synthetic dataset (recipe_data.csv) simulating 180 recipe-interaction records across 40 users, built from five underlying preference archetypes so realistic co-occurrence patterns exist."),
        bullet("Step 2: Load the dataset and explore its shape, columns, and check for missing values."),
        bullet("Step 3: Select the 17 binary ingredient/characteristic columns and prepare the visible-unit feature matrix (no scaling needed since the data is already binary)."),
        bullet("Step 4: Train a BernoulliRBM model with 6 hidden units on the feature matrix."),
        bullet("Step 5: Transform every recipe record into the hidden space to obtain its hidden-unit activation vector."),
        bullet("Step 6: Inspect the learned weight matrix (components_) to find, for each hidden unit, the visible features it is most strongly associated with, and interpret the resulting latent pattern."),
        bullet("Step 7: Group recipe records by known characteristics (Vegetarian, Spicy, Sweet, Healthy) and compare their average hidden-unit activations."),
        bullet("Step 8: Generate visualizations -- dataset preview, hidden-unit heatmap, group comparison bar chart, latent pattern heatmap, and a PCA cluster plot."),

        heading("6. Implementation"),
        heading("Tools and Libraries", HeadingLevel.HEADING_2),
        bullet("pandas: used for loading the CSV dataset and manipulating it as a DataFrame."),
        bullet("numpy: used for numerical array operations such as building the feature matrix and sorting weight values."),
        bullet("matplotlib and seaborn: used to create all professional visualizations, including heatmaps and bar charts."),
        bullet("scikit-learn (BernoulliRBM): used to implement and train the Restricted Boltzmann Machine."),
        bullet("scikit-learn (PCA): used to reduce the 6-dimensional hidden representation to 2 dimensions for cluster visualization."),
        heading("Source Code / GitHub Repository", HeadingLevel.HEADING_2),
        body("The complete source code for this project, including the dataset generator, the RBM analysis script, and the Jupyter notebook, is available at the following repository:"),
        body("[Insert your GitHub repository link here, e.g. https://github.com/your-username/rbm-recipe-preference-project]", { italics: true }),

        heading("7. Results and Output"),
        body("The following visualizations were generated by the project and saved in the results/ folder."),

        image("results/dataset_preview.png", 580, 165),
        caption("Figure 1: Preview of the first 10 records of the recipe dataset."),

        image("results/hidden_unit_heatmap.png", 420, 520),
        caption("Figure 2: Heatmap of hidden-unit activations for a sample of 30 recipe records. Brighter cells indicate stronger activation of that hidden unit for that record."),

        image("results/latent_preference_patterns.png", 580, 300),
        caption("Figure 3: Heatmap of RBM weights connecting each hidden unit to every visible ingredient feature. Warm (red) cells show strong positive association; cool (blue) cells show negative association. This is the primary evidence used to interpret each hidden unit's latent meaning."),

        image("results/user_representation_comparison.png", 580, 300),
        caption("Figure 4: Average hidden-unit activation compared across four preference groups (Vegetarian, Spicy, Sweet, Healthy), each split into their Yes (=1) and No (=0) categories."),

        image("results/preference_clusters.png", 460, 360),
        caption("Figure 5: PCA projection of the 6-dimensional hidden representations into 2 dimensions, colored by whether the recipe is vegetarian, showing how similar preference profiles tend to group together."),

        heading("8. Analysis"),
        body("The learned weight matrix shows that the hidden units did not form randomly -- they organized themselves around intuitive, human-recognizable food-preference themes. For example, certain hidden units consistently received their strongest weights from Sugar, Chocolate, and Sweet, suggesting the RBM discovered a 'sweet/dessert preference' pattern. Other hidden units received strong weights from Chili, Spices, and Spicy, suggesting a 'spicy food preference' pattern. A separate hidden unit aligned with Fruits, Vegetables, and Vegetarian, indicating a 'vegetarian/healthy preference' pattern."),
        body("This structure was further confirmed in Step 7: when recipe records were grouped by known characteristics such as Spicy = 1 versus Spicy = 0, the average activation of the hidden units associated with spice-related features was clearly and consistently higher for the Spicy = 1 group. The same separation was observed for the Sweet and Vegetarian/Healthy groups. This shows that the RBM's latent space, despite being trained without any labels, aligns meaningfully with real, human-understood preference categories."),
        body("The PCA visualization additionally shows that recipe records with similar ingredient profiles tend to occupy nearby regions in the reduced 2-dimensional latent space, providing visual evidence that the RBM's hidden representation preserves meaningful similarity between recipes."),

        heading("9. Conclusion"),
        body("This project successfully demonstrates that a Restricted Boltzmann Machine can learn useful, human-interpretable latent representations from purely unlabeled, binary recipe and ingredient data. Without ever being told what 'vegetarian', 'spicy', or 'sweet' means, the RBM's hidden units naturally organized themselves around these very concepts, based only on how frequently ingredients co-occurred in the data. This confirms the core strength of unsupervised feature learning: the ability to uncover meaningful structure in data automatically. Such an approach could be extended in future work to larger real-world datasets, additional hidden units, or integrated into a full recommendation system for personalized recipe suggestions."),

        heading("10. References"),
        bullet("Scikit-learn Documentation: BernoulliRBM -- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html"),
        bullet("Scikit-learn Documentation: Restricted Boltzmann Machine features -- https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html"),
        bullet("Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation."),
        bullet("Pandas Documentation -- https://pandas.pydata.org/docs/"),
        bullet("NumPy Documentation -- https://numpy.org/doc/"),
        bullet("Matplotlib Documentation -- https://matplotlib.org/stable/contents.html"),
        bullet("Seaborn Documentation -- https://seaborn.pydata.org/"),

        new Paragraph({ children: [new PageBreak()] }),
        heading("Viva Questions and Answers"),

        ...[
          ["What is a Restricted Boltzmann Machine?", "It is a two-layer stochastic neural network (a visible layer and a hidden layer) with no connections within the same layer, used to learn a probability distribution over its input data and discover hidden features."],
          ["Why is RBM called an unsupervised learning model?", "Because it learns patterns directly from the input data without using any labeled output; there is no target variable being predicted."],
          ["What are visible and hidden units?", "Visible units represent the raw input features (here, the ingredient/characteristic columns). Hidden units are the latent neurons that the model learns to represent combinations of those features."],
          ["Why did you use binary input data?", "BernoulliRBM assumes its visible units follow a Bernoulli (0/1) distribution, so binary ingredient-presence data matches the model's assumptions exactly, keeping training stable and results easy to interpret."],
          ["What is latent feature learning?", "It is the process by which a model automatically discovers compact, meaningful hidden representations of data, instead of relying on manually defined features."],
          ["Why did you use BernoulliRBM?", "It is scikit-learn's ready-to-use, well-documented, and beginner-friendly RBM implementation designed specifically for binary input data, making it ideal for this project."],
          ["How do you interpret hidden units?", "By examining the trained weight matrix (components_) and finding which visible features have the strongest (most positive) weights for each hidden unit, then reasoning about what real-world pattern those features represent together."],
          ["What is the difference between supervised and unsupervised learning?", "Supervised learning trains a model using labeled input-output pairs to predict a known target, while unsupervised learning finds hidden patterns or structure in data that has no labels."],
          ["What preprocessing was performed?", "The 17 relevant binary ingredient/characteristic columns were selected and converted into a numeric feature matrix; no scaling was needed since the values were already binary."],
          ["How do you compare user preferences?", "By grouping recipe records according to a known characteristic (e.g., Spicy = 1 vs Spicy = 0) and comparing the average hidden-unit activation values between the two groups."],
          ["What do the heatmaps show?", "One heatmap shows how strongly each hidden unit activates for individual recipe records; another shows the learned weight strength between each hidden unit and each visible ingredient feature, revealing latent patterns."],
          ["What are the limitations of this project?", "The dataset is synthetic rather than real user data, the number of hidden units is small, and RBM training can be sensitive to hyperparameters like learning rate and number of iterations."],
          ["How can this project be improved in the future?", "By using a larger real-world recipe dataset, tuning the number of hidden units, stacking RBMs into a Deep Belief Network, or integrating the learned features into a recommendation system."],
          ["Why did you use PCA?", "PCA reduces the 6-dimensional hidden representation into 2 dimensions so it can be visualized on a scatter plot, making it easier to see whether recipes with similar preferences cluster together."],
          ["Explain the complete workflow of your project.", "Generate a synthetic labeled-free dataset, preprocess it into a binary feature matrix, train a BernoulliRBM, extract hidden-unit activations, interpret the learned weights to understand latent patterns, compare activations across known preference groups, and visualize all findings."],
        ].flatMap(([q, a], i) => [
          new Paragraph({
            children: [new TextRun({ text: `${i + 1}. ${q}`, bold: true })],
            spacing: { before: 160, after: 60 },
          }),
          body(a),
        ]),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("College_Project_Report.docx", buffer);
  console.log("Report generated: College_Project_Report.docx");
});
