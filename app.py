import streamlit as st
import numpy as np
from prediction_service import load_eeg_model, predict_from_eeg_signal, generate_frontend_sample

# --- Page Configuration ---
st.set_page_config(page_title="EEG Brain State Detection", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered EEG Brain State Detection")

# --- Load Model ---
# This is cached so the model is only loaded once
@st.cache_resource
def load_model_cached():
    return load_eeg_model()

model = load_model_cached()

if model is None:
    st.error("🚨 CRITICAL ERROR: Trained model not found! Please run the training pipeline first by executing 'run_training_pipeline.py' in the 'backend' folder.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Input Mode")
input_mode = st.sidebar.selectbox(
    "Choose how to provide EEG readings:",
    ["Simulate Brain State", "Manual Paste"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app uses a trained EEGNet model to classify brain states as 'Positive' or 'Negative' based on synthetic but realistic EEG data.")

# --- MAIN PAGE ---

if input_mode == "Simulate Brain State":
    st.subheader("1. Generate a Simulated EEG Signal")
    st.markdown("Select a brain state to simulate. A realistic EEG signal will be generated for you to copy.")
    
    state_to_simulate = st.selectbox("Select State to Simulate:", ["Positive", "Negative"])
    
    if st.button("Generate EEG Data"):
        state_lower = state_to_simulate.lower()
        generated_data = generate_frontend_sample(state=state_lower)
        st.text_area("Copy the data below and paste it in the 'Manual Paste' section", generated_data, height=150)

elif input_mode == "Manual Paste":
    st.subheader("2. Paste EEG Signal and Predict")
    st.markdown("Paste a comma-separated EEG signal below. It must contain exactly **14,336** numeric values (14 channels * 4 seconds * 256 Hz).")
    
    raw_text = st.text_area("EEG numeric input:", height=150)
    
    if st.button("Analyze and Predict"):
        if not raw_text:
            st.warning("Please paste some EEG data first.")
        else:
            try:
                # Convert text to numpy array
                eeg_data = np.array([float(x.strip()) for x in raw_text.split(',') if x.strip()])
                
                with st.spinner("Analyzing brainwaves with AI model..."):
                    # Call the prediction service
                    result = predict_from_eeg_signal(model, eeg_data)
                
                st.success("Analysis Complete!")
                
                st.markdown("---")
                st.markdown(f"### 🧠 Prediction Result:")
                st.metric(label="Alert Level", value=result["alert_level"])
                
                col1, col2 = st.columns(2)
                col1.metric(label="Primary State Detected", value=result["primary_state"])
                col2.metric(label="Interpretation", value=result["interpretation_word"])
                
                st.metric(label="Model Confidence", value=result["confidence"])
                
                with st.expander("Show Technical Details"):
                    st.code(f"Raw Probabilities: {result['probabilities']}")
                    st.code(f"Input Data Shape: {eeg_data.shape}")

            except ValueError as ve:
                st.error(f"Data Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")