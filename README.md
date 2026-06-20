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
<img width="1500" height="565" alt="image" src="https://github.com/user-attachments/assets/1b0d8b07-c247-4653-94a6-cf318dadeec7" />
<img width="1915" height="832" alt="image" src="https://github.com/user-attachments/assets/b149ac1c-16da-4572-a615-43814a5e1878" />
<img width="1918" height="855" alt="image" src="https://github.com/user-attachments/assets/8fb5ca3d-7442-4369-a958-18d5d13d9fce" />
<img width="1918" height="713" alt="image" src="https://github.com/user-attachments/assets/7e2d6a61-c680-4524-af8a-22d8b3e6aa95" />
<img width="1906" height="867" alt="image" src="https://github.com/user-attachments/assets/a3cd9b9f-d5f5-40e3-b4cc-a25cd447bc83" />
<img width="1868" height="866" alt="image" src="https://github.com/user-attachments/assets/5414bf8d-de06-4fb4-9360-c6ab39c43723" />
<img width="1911" height="845" alt="image" src="https://github.com/user-attachments/assets/5231d7cc-a29b-465e-9808-bfb4b8a62631" />



    
