# Neural Sentinel: AI-Powered EEG Brain State Detection

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_&_Streamlit-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Neural Sentinel is a full-stack deep learning project that demonstrates a real-time brain state classification system using EEG signals. The system classifies a user's brain state as either "Positive" (calm, focused) or "Negative" (stressed, anxious) and provides a human-friendly interpretation.

---

### Demo

![Neural Sentinel Screenshot](https://i.imgur.com/your-screenshot-url.png) 
*A screenshot of the final Streamlit application, showing the sidebar for input, the data generation and input areas, and the final prediction output with interpretation.*

---

## 📋 Key Features

*   **Synthetic Data Generation:** Includes a custom `EEGDataGenerator` that creates a large, realistic dataset for two distinct neuro-physiological states, overcoming the common challenge of data scarcity in EEG research.
*   **State-of-the-Art AI Model:** Implements the **EEGNet** architecture, a specialized Convolutional Neural Network designed to automatically learn spatial and temporal features from raw EEG signals.
*   **Decoupled Full-Stack Architecture:** The project is cleanly separated into a **Backend** (data generation, model training) and a **Frontend** (user interface, prediction), demonstrating a robust and scalable software design pattern.
*   **Interactive Web Interface:** A user-friendly web application built with **Streamlit** provides an intuitive interface for generating, inputting, and analyzing EEG signals.
*   **User-Centric Interpretation Layer:** The system translates the AI's binary "Positive" or "Negative" output into a randomly selected, descriptive word (e.g., "Calm," "Tension") to make the results more intuitive.

---

## 🛠️ Technology Stack

*   **Backend:** Python, TensorFlow (with Keras), Scikit-learn, NumPy
*   **Frontend:** Streamlit
*   **Environment:** Python Virtual Environment (`venv`)

---

## 📂 Project Structure

The project is organized into a clean and understandable structure:
eeg_ai_project/
├── backend/
│ ├── saved_model/ # Stores the final trained AI model
│ ├── eeg_generator.py # The custom synthetic data generator
│ ├── eegnet.py # The EEGNet model architecture definition
│ └── run_training_pipeline.py # The main script to train and save the model
│
├── frontend/
│ ├── prediction_service.py # Loads the model and handles prediction logic
│ └── app.py # The main Streamlit application file
│
└── requirements.txt # All project dependencies
code
Code
---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

*   Python 3.9 or higher installed.
*   Git (for cloning the repository).

### 2. Setup and Installation

**1. Clone the repository:**
```sh
git clone https://github.com/your-username/eeg_ai_project.git
cd eeg_ai_project
2. Create a Python Virtual Environment:
code
Sh
python -m venv venv
3. Activate the Virtual Environment:
On Windows:
code
Sh
.\venv\Scripts\activate
On macOS / Linux:
code
Sh
source venv/bin/activate
(Your terminal prompt should now start with (venv))
4. Install the required libraries:
code
Sh
pip install -r requirements.txt
⚙️ How to Use
The project has two main steps: training the AI model (a one-time process) and running the interactive web application.
Step 1: Train the AI Model (Backend)
This script will generate the dataset, train the EEGNet model, and save the final eeg_model.h5 file. This is a crucial first step.
Run the training pipeline from the root project directory:
code
Sh
python backend/run_training_pipeline.py
This will take several minutes to complete. Wait for the confirmation message: --- TRAINING PIPELINE COMPLETE ---.
Step 2: Run the Frontend Application
Once the model is successfully trained and saved, you can start the interactive Streamlit web app.
Run the Streamlit command from the root project directory:
code
Sh
streamlit run frontend/app.py
Your web browser will automatically open with the application running. If not, navigate to the local URL shown in your terminal (e.g., http://localhost:8501).
🧠 Workflow Explained
Backend (Training Pipeline)
Data Generation: The run_training_pipeline.py script first calls the EEGDataGenerator to create thousands of labeled samples for "Positive" and "Negative" brain states.
Data Preparation: The data is reshaped into the 4D format required by the EEGNet model and split into training and testing sets.
Model Training: The EEGNet model is compiled and trained on the dataset. Its performance is evaluated on the unseen test set to ensure accuracy.
Model Saving: The final, trained model is saved to backend/saved_model/eeg_model.h5.
Frontend (Prediction Service)
Model Loading: When the Streamlit app starts, it loads the eeg_model.h5 file into memory.
User Interaction: The UI provides a "Generate -> Copy -> Paste -> Predict" workflow. The user can generate a sample signal and paste it into a text area.
Prediction: The pasted text is converted to a NumPy array, reshaped, and fed to the loaded model, which returns the prediction probabilities.
Interpretation & Display: The system determines the final state ("Positive" or "Negative") and selects a random descriptive word. This user-friendly result is then displayed in the web interface.
