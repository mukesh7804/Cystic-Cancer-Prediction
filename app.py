import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import re
from PIL import Image

# IMPORTANT: Ensure tensorflow is installed in your environment
import tensorflow as tf

# =========================
# PAGE CONFIG & STYLING
# =========================
st.set_page_config(
    page_title="OncoScan AI | Cancer Prediction", 
    page_icon="🩺", 
    layout="wide"
)

# Custom CSS for a professional "Healthcare Tech" look
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #ffffff; }
    h1, h2, h3 { color: #1f1f1f; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL ASSETS
# =========================
@st.cache_resource
def load_assets():
    # Load Tabular Models (Updated to load 'ensemble.pkl' from your recent notebook)
    try:
        tabular_model = pickle.load(open('ensemble.pkl', 'rb')) # Changed from model.pkl based on your notebook
    except FileNotFoundError:
        tabular_model = pickle.load(open('model.pkl', 'rb')) # Fallback
        
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    columns = pickle.load(open('columns.pkl', 'rb'))
    
    # Load Image Model (MobileNetV2)
    try:
        # Assuming you save your image model as 'image_model.h5' or '.keras'
        image_model = tf.keras.models.load_model('image_model.h5')
    except Exception as e:
        image_model = None
        st.sidebar.warning("⚠️ Image model 'image_model.h5' not found. Image classification will be disabled.")
        
    return tabular_model, scaler, columns, image_model

tabular_model, scaler, columns, image_model = load_assets()

# =========================
# PDF GENERATION FUNCTION
# =========================
def generate_pdf(inputs=None, prediction=None, prob=None, mode="Tabular"):
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Diagnostic AI Prediction Report", styles['Title']))
    content.append(Spacer(1, 20))

    result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"
    content.append(Paragraph(f"<b>Prediction Result:</b> {result}", styles['Normal']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"<b>Diagnostic Confidence:</b> {prob*100:.2f}%", styles['Normal']))
    content.append(Spacer(1, 20))

    if mode == "Tabular" and inputs is not None:
        content.append(Paragraph("<b>Input Feature Analysis:</b>", styles['Heading2']))
        content.append(Spacer(1, 10))
        for i, val in enumerate(inputs):
            content.append(Paragraph(f"Feature {i+1} ({columns[i]}): {val}", styles['Normal']))
    elif mode == "Image":
        content.append(Paragraph("<b>Analysis Type:</b> Deep Learning Image Scan (MobileNetV2)", styles['Heading2']))
        content.append(Spacer(1, 10))
        content.append(Paragraph("The visual features of the provided medical scan were analyzed using a Convolutional Neural Network.", styles['Normal']))

    doc.build(content)
    return file_path

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("Control Panel")
    
    analysis_type = st.selectbox("Select Analysis Type", ["Tabular Clinical Data", "Medical Image Scan"])
    
    st.markdown("---")

# =========================
# MAIN UI: TABULAR DATA
# =========================
if analysis_type == "Tabular Clinical Data":
    st.title("🩺 OncoScan AI: Clinical Data Analysis")
    st.markdown("Automated classification of breast mass features using an Ensemble AI Model.")
    st.divider()

    with st.sidebar:
        mode = st.radio("Input Method", ["Slider Input", "Manual Input", "CSV Upload"])
        st.subheader("🧪 Quick Test Samples")
        
        cancer_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        normal_sample = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]

        if 'current_inputs' not in st.session_state:
            st.session_state.current_inputs = [0.0]*30

        if st.button("🔴 Load Malignant Sample"):
            st.session_state.current_inputs = cancer_sample
            st.rerun()

        if st.button("🟢 Load Benign Sample"):
            st.session_state.current_inputs = normal_sample
            st.rerun()

    inputs = st.session_state.current_inputs

    if mode == "Slider Input":
        st.subheader("Fine-Tune Clinical Features")
        cols = st.columns(3)
        for i in range(30):
            with cols[i % 3]:
                inputs[i] = st.slider(f"{columns[i]}", 0.0, 3000.0, float(inputs[i]), key=f"slider_{i}")

    elif mode == "Manual Input":
        st.subheader("Bulk Data Entry")
        user_input = st.text_area("Paste 30 space/comma separated values:", height=150)
        if user_input:
            try:
                cleaned = re.sub(r"[^\d.\-\s,]", " ", user_input)
                inputs = [float(v) for v in cleaned.replace(",", " ").split()]
                if len(inputs) == 30:
                    st.success(f"Successfully loaded {len(inputs)} features.")
                else:
                    st.warning(f"Feature count mismatch: Expected 30, found {len(inputs)}.")
            except:
                st.error("Invalid numeric format detected.")

    else: # CSV Upload
        st.subheader("Dataset Batch Upload")
        file = st.file_uploader("Upload CSV (Must have 30 feature columns)")
        if file:
            df = pd.read_csv(file)
            if df.shape[1] >= 30:
                inputs = df.iloc[0, :30].tolist()
                st.success("CSV features mapped successfully.")
            else:
                st.error(f"Invalid dimensions: CSV has {df.shape[1]} columns. Need 30.")

    st.divider()
    predict_col, result_col = st.columns([1, 2])

    with predict_col:
        st.write("### Ready for Diagnosis?")
        run_prediction = st.button("🚀 Run Clinical Analysis")

    with result_col:
        if run_prediction:
            if len(inputs) != 30:
                st.error("Prediction halted: Incomplete feature set (30 required).")
            else:
                input_df = pd.DataFrame([inputs], columns=columns)
                input_scaled = scaler.transform(input_df)
                
                # Updated for Ensemble Soft Voting (Probabilities instead of decision function)
                prediction = tabular_model.predict(input_scaled)
                try:
                    prob = tabular_model.predict_proba(input_scaled)[0][1] # Probability of Class 1 (Malignant)
                except AttributeError:
                    prob = tabular_model.decision_function(input_scaled)[0] # Fallback for pure SVM

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if prediction[0] == 1:
                        st.metric("Diagnosis", "MALIGNANT", delta="Cancer Detected", delta_color="inverse")
                    else:
                        st.metric("Diagnosis", "BENIGN", delta="No Cancer Detected")
                
                with m_col2:
                    st.metric("Malignancy Probability", f"{prob*100:.2f}%")

                st.write("### Diagnostic Confidence Visualization")
                fig, ax = plt.subplots(figsize=(8, 2))
                color = '#ff4b4b' if prediction[0] == 1 else '#00cc96'
                
                # Updated Chart to show 0 to 1 Probability scale
                ax.barh(["Risk Probability"], [prob], color=color)
                ax.set_xlim(0, 1) 
                ax.axvline(0.5, color='black', linestyle='--')
                ax.text(0.1, 0, 'Benign Zone', fontsize=10, color='black')
                ax.text(0.6, 0, 'Malignant Zone', fontsize=10, color='black')
                st.pyplot(fig)

                pdf_file = generate_pdf(inputs, prediction[0], prob, mode="Tabular")
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="📄 Download Official Medical Report",
                        data=f, file_name="tabular_report.pdf", mime="application/pdf"
                    )

# =========================
# MAIN UI: IMAGE SCAN
# =========================
elif analysis_type == "Medical Image Scan":
    st.title("🔬 OncoScan AI: Medical Image Analysis")
    st.markdown("Upload ultrasound, MRI, or histopathology scans for MobileNetV2 Deep Learning analysis.")
    st.divider()

    if image_model is None:
        st.error("⚠️ The Image Neural Network model is not loaded. Please ensure 'image_model.h5' is in your project folder.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Upload Scan")
            img_file = st.file_uploader("Choose a Medical Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
            
            if img_file is not None:
                img = Image.open(img_file).convert('RGB')
                st.image(img, caption="Uploaded Scan", use_container_width=True)
                
        with col2:
            st.write("### Analysis Engine")
            if img_file is not None:
                if st.button("🚀 Run Image Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image features..."):
                        # 1. Resize to MobileNetV2 expected size
                        img_resized = img.resize((224, 224))
                        
                        # 2. Convert to array and apply MobileNetV2 Preprocessing
                        img_array = np.array(img_resized)
                        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                        
                        # 3. Expand dimensions to create a batch of 1: shape (1, 224, 224, 3)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # 4. Predict
                        preds = image_model.predict(img_array)[0]
                        
                        # Assuming Class 0 is Benign, Class 1 is Malignant based on standard binary tasks
                        # Update these indices if your model outputs them in reverse
                        pred_class = np.argmax(preds)
                        confidence = preds[pred_class]
                        
                        st.subheader("Results")
                        if pred_class == 1:
                            st.error(f"🚨 **MALIGNANT** (Confidence: {confidence*100:.2f}%)")
                        else:
                            st.success(f"✅ **BENIGN** (Confidence: {confidence*100:.2f}%)")
                            
                        # Chart
                        st.write("### Prediction Confidence")
                        fig, ax = plt.subplots(figsize=(8, 2))
                        colors = ['#00cc96', '#ff4b4b'] # Green for 0, Red for 1
                        ax.barh(["Benign", "Malignant"], [preds[0], preds[1]], color=colors)
                        ax.set_xlim(0, 1)
                        st.pyplot(fig)
                        
                        # PDF Download
                        pdf_file = generate_pdf(prediction=pred_class, prob=confidence, mode="Image")
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="📄 Download Image Scan Report",
                                data=f, file_name="image_report.pdf", mime="application/pdf"
                            )

st.divider()
st.caption("Disclaimer: This tool is intended for research and educational purposes. All AI predictions must be verified by a certified healthcare professional.")