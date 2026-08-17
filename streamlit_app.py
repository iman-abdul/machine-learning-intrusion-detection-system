
import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "binary_hgb_model.pkl"
THRESHOLD = 0.80

# Load trained IDS model
model = joblib.load(MODEL_PATH)

# Get the exact features expected by the model
EXPECTED_FEATURES = list(
    model.named_steps["preprocessor"].feature_names_in_
)


st.set_page_config(
    page_title="Machine Learning IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Machine Learning Intrusion Detection System")

st.write(
    "Upload a network-flow CSV file. "
    "The trained HistGradientBoosting model "
    "will classify each flow as Normal or Attack."
)

uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write(
            f"Input data shape: {data.shape}"
        )

        # Check required features
        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in data.columns
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        else:

            # Arrange features in the exact order
            # expected by the trained pipeline
            model_input = data[
                EXPECTED_FEATURES
            ].copy()

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

            # Create output
            results = model_input.copy()

            results["attack_probability"] = probabilities
            results["prediction"] = predictions

            # Summary statistics
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

            # Display summary
            st.subheader("Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Flows",
                total
            )

            col2.metric(
                "Normal Flows",
                normal
            )

            col3.metric(
                "Attack Flows",
                attacks
            )

            col4.metric(
                "Attack Percentage",
                f"{attack_percentage:.2f}%"
            )

            st.write(
                f"Detection threshold: **{THRESHOLD:.2f}**"
            )

            # Display predictions
            st.subheader("IDS Predictions")

            st.dataframe(
                results,
                use_container_width=True
            )

            # Download results
            csv_output = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="ids_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:

        st.error(
            f"Error processing file: {error}"
        )
