# 1. Project Title
Tomato Leaf Fungal vs Bacterial Classification with Gene-Informed Biological Interpretation

# 2. Problem Statement
This project solves binary disease-pattern classification for tomato leaf images using deep learning.
The goal is to distinguish fungal vs bacterial patterns in a way that is accurate, reproducible, and explainable for academic oral defense.
Tomato disease classification matters because early category-level detection can support faster field-level decision-making.

# 3. Project Scope
- Tomato-only image scope
- Two classes only:
  - `bacterial`
  - `fungal`
- Removed from final scope:
  - viral class
  - stress class
  - non-tomato images

# 4. Dataset
- Source type: PlantVillage-style real leaf disease image dataset (curated and cleaned for this project scope).
- Final split counts:
  - train: bacterial = 1488, fungal = 4925, total = 6413
  - val: bacterial = 319, fungal = 1055, total = 1374
  - test: bacterial = 320, fungal = 1056, total = 1376
- Final preprocessing and integrity steps:
  - cleaned to tomato-only bacterial/fungal classes
  - removed irrelevant old-scope branches
  - rebuilt split using fixed seed
  - applied hash-based grouping to prevent duplicate-content leakage across train/val/test

# 5. Model
- Framework: PyTorch
- Backbone: ResNet18 (transfer learning)
- Adaptation:
  - final fully connected layer replaced for 2 outputs
  - light fine-tuning on `layer4` + `fc`
- Training details:
  - weighted `CrossEntropyLoss` for class imbalance
  - optimizer: Adam
  - learning-rate scheduling: ReduceLROnPlateau on validation loss

# 6. Results
- Test accuracy: `0.993459`
- Macro F1-score: `0.990926`
- Confusion matrix (rows = true, cols = predicted):
  - bacterial: 320 correct, 0 misclassified
  - fungal: 1047 correct, 9 misclassified
- Short interpretation:
  - The final model shows very strong fungal-vs-bacterial separation on the cleaned test split, with small residual fungal-to-bacterial confusion.

# 7. Biological Interpretation Layer
- File: `data/gene_interpretation.json`
- Purpose:
  - The model predicts only class label (`bacterial` or `fungal`).
  - Gene entries are curated tomato biological interpretation mapped from the predicted class.
- Important disclaimer:
  - Genes are not predicted from the image.
  - They are an interpretation layer to improve biological explainability.

# 8. Streamlit Demo
The dashboard allows a user to:
- upload one tomato leaf image
- run trained-model inference
- view predicted class and confidence scores
- view curated class-based gene interpretation and disclaimer
- view static dataset-level performance references (training curves and confusion matrix)

# 9. Project Structure
```text
data/
scripts/
src/
app/
outputs/
README.md
requirements.txt
```

# 10. How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

# 11. Limitations
- Class imbalance exists (fungal is the majority class).
- Dataset scope is restricted to tomato images only.
- Gene information is curated interpretation, not model-predicted output.
- Model confidence is classification confidence, not biological certainty.

# 12. Conclusion
The project delivers an end-to-end, explainable pipeline for tomato fungal-vs-bacterial classification, with strong test performance and a clear curated biological interpretation layer for final presentation and demonstration.
