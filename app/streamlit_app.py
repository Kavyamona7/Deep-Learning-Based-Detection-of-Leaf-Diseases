from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Allow importing from src/ when running: streamlit run app/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretation import build_interpretation, load_gene_data  # noqa: E402


def get_paths() -> Dict[str, Path]:
    return {
        "checkpoint": PROJECT_ROOT / "outputs" / "models" / "best_resnet18.pth",
        "gene_json": PROJECT_ROOT / "data" / "gene_interpretation.json",
        "phase3_summary": PROJECT_ROOT / "outputs" / "logs" / "phase3_summary.txt",
        "training_curves": PROJECT_ROOT / "outputs" / "plots" / "training_curves.png",
        "confusion_matrix": PROJECT_ROOT / "outputs" / "plots" / "confusion_matrix.png",
    }


def validate_class_mapping(class_to_idx: Dict[str, int]) -> None:
    expected = {"bacterial": 0, "fungal": 1}
    if class_to_idx != expected:
        raise ValueError(
            f"class_to_idx mismatch. Expected {expected}, got {class_to_idx}"
        )

    idx_values = sorted(class_to_idx.values())
    if idx_values != [0, 1]:
        raise ValueError(
            f"class_to_idx values must be [0, 1], got {idx_values}"
        )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@st.cache_resource
def load_model_and_mapping(checkpoint_path: Path) -> Tuple[nn.Module, Dict[str, int], Dict[int, str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict):
        raise ValueError("Checkpoint does not contain a valid class_to_idx mapping.")

    validate_class_mapping(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_to_idx, idx_to_class


@st.cache_data
def load_phase3_summary(summary_path: Path) -> str:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return summary_path.read_text(encoding="utf-8")


def predict_single_image(
    image: Image.Image,
    model: nn.Module,
    idx_to_class: Dict[int, str],
) -> Tuple[str, float, Dict[str, float]]:
    transform = build_eval_transform()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    predicted_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx].item())

    class_scores = {idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))}
    return predicted_class, confidence, class_scores


def render_class_scores(class_scores: Dict[str, float]) -> None:
    st.write("**Class Confidence Scores**")
    st.caption("Confidence reflects model classification confidence, not biological certainty.")

    for class_name in ("bacterial", "fungal"):
        score = class_scores.get(class_name, 0.0)
        st.write(f"{class_name.capitalize()}: **{score * 100:.2f}%**")
        st.progress(score)


def main() -> None:
    st.set_page_config(
        page_title="Leaf Fungal vs Bacterial Classification",
        layout="wide",
    )

    st.title("Leaf Fungal vs Bacterial Classification")
    st.caption(
        "The model predicts bacterial vs fungal class only; gene information is curated biological interpretation."
    )

    paths = get_paths()

    # Preload required assets with clear error messages.
    try:
        model, class_to_idx, idx_to_class = load_model_and_mapping(paths["checkpoint"])
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop()

    try:
        gene_data = load_gene_data(paths["gene_json"])
    except Exception as e:
        st.error(f"Gene data loading error: {e}")
        st.stop()

    try:
        phase3_summary = load_phase3_summary(paths["phase3_summary"])
    except Exception as e:
        st.error(f"Phase 3 summary loading error: {e}")
        st.stop()

    # -------------------------
    # Top section: upload + preview (left), prediction + scores (right)
    # -------------------------
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a tomato leaf image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is None:
            st.warning("Please upload one tomato leaf image (jpg/jpeg/png) to run prediction.")
            st.caption(f"Current class mapping: {class_to_idx}")
            image = None
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error("Could not read this image file. Please upload a valid jpg/jpeg/png file.")
                st.stop()

            st.subheader("Uploaded Image Preview")
            st.image(image, caption="Uploaded leaf image", width=380)

    with right_col:
        st.subheader("Prediction Result (Uploaded Image)")
        st.caption("This result is based on the uploaded leaf image.")

        if uploaded_file is None:
            st.info("Upload an image to view prediction and confidence scores.")
            predicted_class = None
            confidence = None
            class_scores = None
        else:
            try:
                predicted_class, confidence, class_scores = predict_single_image(
                    image=image,
                    model=model,
                    idx_to_class=idx_to_class,
                )
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Predicted Class", predicted_class.capitalize())
            with metric_col2:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            render_class_scores(class_scores)

    # -------------------------
    # Biological interpretation (secondary)
    # -------------------------
    if uploaded_file is not None:
        st.subheader("Biological Interpretation (Curated, Class-Based)")
        try:
            interpretation = build_interpretation(
                predicted_class=predicted_class,
                confidence=confidence,
                gene_data=gene_data,
            )
        except Exception as e:
            st.error(f"Interpretation error: {e}")
            st.stop()

        st.markdown(f"**{interpretation['biological_interpretation_title']}**")
        st.write(interpretation["short_explanation"])

        st.write("**Associated Genes**")
        genes_table = [
            {
                "gene": item["gene"],
                "role": item["role"],
                "description": item["description"],
                "source": item["source"],
            }
            for item in interpretation["associated_genes"]
        ]

        try:
            st.dataframe(genes_table, use_container_width=True, hide_index=True)
        except TypeError:
            st.table(genes_table)

        st.info(interpretation["disclaimer"])

    # -------------------------
    # Tertiary references: summary + static plots
    # -------------------------
    st.markdown("---")
    st.write("A concise project summary and final evaluation notes are available below.")
    with st.expander("View detailed project summary", expanded=False):
        st.text(phase3_summary)

    st.subheader("Model Performance Reference (dataset-level)")
    st.caption(
        "These are static dataset-level results, not evidence specific to the uploaded image."
    )

    tab_curves, tab_cm = st.tabs(["Training Curves", "Confusion Matrix"])

    with tab_curves:
        if paths["training_curves"].exists():
            st.image(
                str(paths["training_curves"]),
                caption="Training Curves (dataset-level reference)",
                width=760,
            )
        else:
            st.error(f"Missing plot file: {paths['training_curves']}")

    with tab_cm:
        if paths["confusion_matrix"].exists():
            st.image(
                str(paths["confusion_matrix"]),
                caption="Confusion Matrix (dataset-level reference)",
                width=760,
            )
        else:
            st.error(f"Missing plot file: {paths['confusion_matrix']}")


if __name__ == "__main__":
    main()
