import os
import numpy as np
import tensorflow as tf
import random
import sys

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from eeg_generator import EEGDataGenerator

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'saved_model', 'eeg_model.h5')
CLASS_LABELS = {0: "Positive", 1: "Negative"}

# Your new interpretation layer!
POSITIVE_WORDS = ["Calm", "Peaceful", "Happy", "Pleasant", "Focused", "Engaged", "Resting", "Neutral", "Confident", "Clear-minded"]
NEGATIVE_WORDS = ["Tension", "Anxiety", "Anger", "Frustration", "Sad", "Depressed", "Fatigue", "Drowsy", "Confused", "Distracted"]

# --- Load Model ---
def load_eeg_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# --- Prediction Function ---
def predict_from_eeg_signal(model, eeg_data):
    """Takes raw EEG data and returns a full prediction dictionary."""
    if model is None:
        raise ValueError("Model is not loaded. Please train the model first.")

    # Reshape for the model: (1, channels, timesteps, 1)
    n_channels = 14
    n_timesteps = 256 * 4
    if eeg_data.size != (n_channels * n_timesteps):
        raise ValueError(f"Input data must have {n_channels * n_timesteps} values.")
        
    sample_reshaped = eeg_data.reshape(1, n_channels, n_timesteps, 1)

    # Make prediction
    probs = model.predict(sample_reshaped)[0]
    pred_index = np.argmax(probs)
    pred_label = CLASS_LABELS[pred_index]
    confidence = np.max(probs) * 100

    # Your new interpretation logic
    if pred_label == "Positive":
        interpretation_word = random.choice(POSITIVE_WORDS)
        alert_level = "🟢 NORMAL"
    else:
        interpretation_word = random.choice(NEGATIVE_WORDS)
        alert_level = "🔴 ATTENTION"
    
    return {
        "alert_level": alert_level,
        "primary_state": pred_label,
        "interpretation_word": interpretation_word,
        "confidence": f"{confidence:.2f}%",
        "probabilities": f"Positive: {probs[0]:.3f}, Negative: {probs[1]:.3f}"
    }

# --- Data Generation for Frontend ---
def generate_frontend_sample(state='positive'):
    """Generates a sample and formats it as a string for the UI."""
    generator = EEGDataGenerator()
    sample = generator.generate_eeg_sample(state)
    # Convert to a comma-separated string
    return ", ".join([f"{val:.4f}" for val in sample.flatten()])