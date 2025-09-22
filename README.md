🩺 Breast Cancer Prediction using Deep Learning

This project implements a Neural Network model using TensorFlow/Keras to predict whether a tumor is benign or malignant based on the Breast Cancer Wisconsin dataset.

📌 Project Overview

Objective: Classify breast cancer tumors (benign/malignant) using deep learning.

Dataset: Breast Cancer Wisconsin Diagnostic Dataset (30 features).

Approach: Neural Network with TensorFlow/Keras.

Outcome: Achieved high accuracy in detecting cancer with minimal preprocessing.

⚙️ Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn (for visualization)

Scikit-learn (for preprocessing & evaluation)

🧠 Model Architecture
Input Layer: 30 features (Flattened)
Hidden Layer: Dense(16), activation = ReLU
Output Layer: Dense(2), activation = Sigmoid

📊 Workflow

Data Loading: Breast Cancer dataset from sklearn.datasets.

Preprocessing:

Feature scaling (normalization).

Train-test split.

Model Building: Sequential Neural Network.

Model Training: Optimized with Adam optimizer & categorical cross-entropy loss.

Evaluation: Accuracy

🚀 Results

Training Accuracy: ~99%

Test Accuracy: ~98%


📈 Visualizations

Accuracy vs Epochs

Loss vs Epochs

Confusion Matrix

🔮 Future Improvements

Experiment with deeper networks (more hidden layers).

Apply regularization (Dropout, L2) to avoid overfitting.

Try advanced models: Random Forest, XGBoost, or CNNs.

Hyperparameter tuning using GridSearchCV or Keras Tuner.
