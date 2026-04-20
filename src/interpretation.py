from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


EXPECTED_TOP_LEVEL_KEYS = {"bacterial", "fungal", "disclaimer"}
EXPECTED_GENE_FIELDS = {"gene", "plant", "role", "description", "source"}


def load_gene_data(json_path: Path | None = None) -> Dict[str, Any]:
    if json_path is None:
        project_root = Path(__file__).resolve().parents[1]
        json_path = project_root / "data" / "gene_interpretation.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing gene interpretation file: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        gene_data = json.load(f)

    validate_gene_data(gene_data)
    return gene_data


def validate_gene_data(gene_data: Dict[str, Any]) -> None:
    if not isinstance(gene_data, dict):
        raise ValueError("Gene data must be a JSON object.")

    keys = set(gene_data.keys())
    if keys != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            f"Top-level keys must be exactly {sorted(EXPECTED_TOP_LEVEL_KEYS)}. Found: {sorted(keys)}"
        )

    for category in ("bacterial", "fungal"):
        entries = gene_data.get(category)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"'{category}' must be a non-empty list.")

        for idx, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{category}[{idx}] must be an object.")
            missing = EXPECTED_GENE_FIELDS - set(entry.keys())
            if missing:
                raise ValueError(f"{category}[{idx}] missing fields: {sorted(missing)}")

    disclaimer = gene_data.get("disclaimer")
    if not isinstance(disclaimer, dict) or "text" not in disclaimer:
        raise ValueError("The 'disclaimer' section must be an object with a 'text' field.")


def _short_explanation(predicted_class: str) -> str:
    if predicted_class == "fungal":
        return (
            "The model predicts a fungal disease pattern in this tomato leaf. "
            "The associated genes shown below are curated tomato defense-related genes linked to "
            "fungal pathogen recognition, signaling, and response."
        )
    if predicted_class == "bacterial":
        return (
            "The model predicts a bacterial disease pattern in this tomato leaf. "
            "The associated genes shown below are curated tomato immune genes linked to bacterial "
            "recognition and downstream defense signaling."
        )
    raise ValueError("predicted_class must be 'bacterial' or 'fungal'.")


def build_interpretation(
    predicted_class: str, confidence: float, gene_data: Dict[str, Any]
) -> Dict[str, Any]:
    validate_gene_data(gene_data)

    if predicted_class not in ("bacterial", "fungal"):
        raise ValueError("predicted_class must be 'bacterial' or 'fungal'.")
    if not isinstance(confidence, (float, int)):
        raise ValueError("confidence must be numeric.")

    confidence = float(confidence)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "biological_interpretation_title": f"Tomato {predicted_class.capitalize()} Interpretation",
        "associated_genes": gene_data[predicted_class],
        "short_explanation": _short_explanation(predicted_class),
        "disclaimer": gene_data["disclaimer"]["text"],
    }


if __name__ == "__main__":
    data = load_gene_data()

    fungal_example = build_interpretation("fungal", 0.93, data)
    bacterial_example = build_interpretation("bacterial", 0.88, data)

    print("Fungal sanity example:")
    print(json.dumps(fungal_example, indent=2))
    print("\nBacterial sanity example:")
    print(json.dumps(bacterial_example, indent=2))
