import os
import numpy as np
from sklearn.model_selection import train_test_split
from eeg_generator import EEGDataGenerator
from eegnet import EEGNet

# --- Configuration ---
N_CHANNELS = 14
DURATION = 4
FS = 256
N_SAMPLES_PER_CLASS = 1000
N_TIMESTEPS = DURATION * FS
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved_model/')
MODEL_NAME = 'eeg_model.h5'

# --- PHASE 1: DATA FOUNDATION ---
print("--- [PHASE 1] Starting Data Generation ---")
generator = EEGDataGenerator(n_channels=N_CHANNELS, duration=DURATION, fs=FS)
X_data, y_data = generator.generate_dataset(n_samples_per_class=N_SAMPLES_PER_CLASS)

# --- PHASE 2: DATA PREPARATION ---
print("\n--- [PHASE 2] Starting Data Preparation ---")
X_reshaped = X_data.reshape(X_data.shape[0], N_CHANNELS, N_TIMESTEPS, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# --- PHASE 3: MODEL TRAINING ---
print("\n--- [PHASE 3] Starting Model Training ---")
model = EEGNet(nb_classes=2, Chans=N_CHANNELS, Samples=N_TIMESTEPS, dropoutRate=0.5)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(
    X_train, y_train, batch_size=32, epochs=25, validation_split=0.2
)

# --- PHASE 4: EVALUATION ---
print("\n--- [PHASE 4] Starting Final Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# --- PHASE 5: SAVING THE MODEL ---
print("\n--- [PHASE 5] Saving the Trained Model ---")
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
print(f"Model saved to: {os.path.join(MODEL_SAVE_PATH, MODEL_NAME)}")
print("\n--- TRAINING PIPELINE COMPLETE ---")