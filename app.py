import streamlit as st
from image_classifier import ImageClassification
from insights import get_insights_stream, detect_breed
import pandas as pd
from PIL import Image
import time
import io

# Page configuration
st.set_page_config(
    page_title="Dairy Animal Analyzer",
    page_icon="🐄",
    layout="wide"
)

# Initialize session state variables
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "analyzer_image" not in st.session_state:
    st.session_state.analyzer_image = None

# Sidebar menu
with st.sidebar:
    selected = st.radio(
        "Select the mode",['Image Classification','The Dairy Analyzer']
    )

# ---------------- IMAGE CLASSIFICATION ----------------
if selected == 'Image Classification':
    st.header("🐄 Cow/Buffalo Image Classification")

    col1, col2 = st.columns([2, 1], border=True)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image of a cow or buffalo",
            type=["jpg", "jpeg", "png"],
            key="classification_upload"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.session_state.uploaded_image is not None:
            if st.button("Classify Animal", type="primary"):
                with st.spinner("Classifying..."):
                    classifier = ImageClassification()
                    # Convert in-memory image to bytes for classifier
                    img_bytes = io.BytesIO()
                    st.session_state.uploaded_image.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)

                    result = classifier.image_classification(img_bytes)

                    st.success(f"Classification Result: **{result.capitalize()}**")

                    if hasattr(classifier, 'last_confidence'):
                        st.metric(
                            label="Confidence",
                            value=f"{classifier.last_confidence:.2%}"
                        )

# ---------------- DAIRY ANALYZER ----------------
elif selected == 'The Dairy Analyzer':
    st.header("📊 The Dairy Analyzer")

    col1, col2 = st.columns([1, 2], border=True)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image for analysis",
            type=["jpg", "jpeg", "png"],
            key="analyzer_upload"
        )

        if uploaded_file is not None:
            st.session_state.analyzer_image = Image.open(uploaded_file).convert("RGB")
            st.image(st.session_state.analyzer_image, caption="Uploaded Image", use_container_width=True)

            animal_type = st.radio(
                "Select animal type:",
                ["Cow", "Buffalo"],
                horizontal=True,
                key="animal_type"
            )

    with col2:
        if st.session_state.analyzer_image is not None and st.button("Analyze Animal", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            insights_data = None

            # Convert image to bytes
            img_bytes = io.BytesIO()
            st.session_state.analyzer_image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            for update in get_insights_stream(animal_type.lower(), img_bytes):
                progress_bar.progress(update['progress'] / 100)
                status_text.text(update['message'])

                if 'data' in update:
                    insights_data = update['data']

                time.sleep(0.5)

            if insights_data:
                status_text.empty()
                progress_bar.empty()

                st.subheader(f"Deep Analyzer for this {animal_type}")

                insights_df = pd.DataFrame([
                    {"Metric": "Breed Type", "Value": insights_data["breed_type"]},
                    {"Metric": "Starting Expenditure", "Value": insights_data["starting_expenditure"]},
                    {"Metric": "Monthly Income", "Value": insights_data["monthly_income"]},
                    {"Metric": "Annual Income", "Value": insights_data["annual_income"]},
                    {"Metric": "Milk Production (per day)", "Value": insights_data["milk_per_day"]},
                    {"Metric": "Popular Areas", "Value": insights_data["popular_areas"]},
                    {"Metric": "Farmers Percentage", "Value": insights_data["farmers_percent"]}
                ])

                st.dataframe(
                    insights_df,
                    column_config={
                        "Metric": st.column_config.Column(width="medium"),
                        "Value": st.column_config.Column(width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )


