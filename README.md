# OncoScan AI 🩺

OncoScan AI is a dual-modal web application designed to assist in cancer prediction. Built with Streamlit, it leverages both traditional Machine Learning for clinical data and Deep Learning for medical image scans to provide predictive diagnostics and confidence scores.

> **Disclaimer:** This tool is intended for research and educational purposes only. All AI predictions must be verified by a certified healthcare professional.

## ✨ Features

* **Tabular Data Analysis:** Predicts malignancy based on clinical features (like radius, texture, perimeter, and area) using an advanced Scikit-Learn Ensemble model (SVM + Random Forest).
* **Medical Image Scanning:** Analyzes medical images/scans (e.g., cysts) using a fine-tuned MobileNetV2 Convolutional Neural Network (CNN) via TensorFlow/Keras.
* **Instant Diagnostic Reports:** Automatically generates downloadable PDF prediction reports using ReportLab.
* **Interactive UI:** Built with Streamlit for a clean, professional, and accessible user experience.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Deep Learning:** TensorFlow, Keras (MobileNetV2)
* **Reporting:** ReportLab
* **Visualization:** Matplotlib

📂 Repository Structure

    app.py: The main Streamlit web application.

    cyst_cancer.ipynb: Jupyter notebook containing the model training and data exploration processes.

    data.csv: The primary dataset used for training the tabular model.

    check_files.py: Utility script to verify the integrity of the pickle files.

Model Assets:

    ensemble.pkl, model.pkl, scaler.pkl, columns.pkl: Saved models, scalers, and feature columns for tabular predictions.

    image_model.h5: The trained Deep Learning model for image predictions.

📸 Screenshots
<img width="1902" height="797" alt="image" src="https://github.com/user-attachments/assets/1b0d8b07-c247-4653-94a6-cf318dadeec7" />


    
