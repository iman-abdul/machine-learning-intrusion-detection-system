
import gradio as gr
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "binary_hgb_model.pkl"
THRESHOLD = 0.80

# Load the trained pipeline once when the app starts
model = joblib.load(MODEL_PATH)

# Get the exact raw features expected by the trained pipeline
EXPECTED_FEATURES = list(
    model.named_steps["preprocessor"].feature_names_in_
)


def predict_network_traffic(file):
    """
    Analyze uploaded network-flow CSV data
    and classify each flow as Normal or Attack.
    """

    if file is None:
        return None, "Please upload a CSV file."

    try:
        # Load uploaded CSV
        data = pd.read_csv(file)

        # Check required columns
        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in data.columns
        ]

        if missing_features:
            return (
                None,
                "Missing required features:\n"
                + ", ".join(missing_features)
            )

        # Select features in the exact order expected by the model
        model_input = data[EXPECTED_FEATURES].copy()

        # Generate attack probabilities
        probabilities = model.predict_proba(
            model_input
        )[:, 1]

        # Apply optimized threshold
        predictions = np.where(
            probabilities >= THRESHOLD,
            "Attack",
            "Normal"
        )

        # Build output
        results = model_input.copy()

        results["attack_probability"] = probabilities
        results["prediction"] = predictions

        # Calculate summary
        total = len(results)
        attacks = int(
            (predictions == "Attack").sum()
        )
        normal = int(
            (predictions == "Normal").sum()
        )

        attack_percentage = (
            attacks / total * 100
            if total > 0
            else 0
        )

        summary = (
            f"Total flows: {total}\n"
            f"Normal flows: {normal}\n"
            f"Attack flows: {attacks}\n"
            f"Attack percentage: {attack_percentage:.2f}%\n"
            f"Detection threshold: {THRESHOLD:.2f}"
        )

        return results, summary

    except Exception as error:
        return None, f"Error: {error}"


demo = gr.Interface(
    fn=predict_network_traffic,
    inputs=gr.File(
        label="Upload Network Traffic CSV",
        type="filepath"
    ),
    outputs=[
        gr.Dataframe(
            label="IDS Predictions"
        ),
        gr.Textbox(
            label="Detection Summary"
        )
    ],
    title="Machine Learning Intrusion Detection System",
    description=(
        "Upload a CSV containing the required "
        "network-flow features. The trained "
        "HistGradientBoosting IDS will classify "
        "each flow as Normal or Attack."
    )
)


if __name__ == "__main__":
    demo.launch()
