import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fracture AI | Clinical Suite", layout="wide", page_icon="🏥")

if 'executed' not in st.session_state:
    st.session_state.executed = False
    st.session_state.severity = ""
    st.session_state.details = {}

# ===============================
# UI THEME
# ===============================
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    .stApp { background: #f1f5f9; color: #1e293b; }
    .hero-section {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 25px 10px; border-radius: 15px; text-align: center; 
        margin-bottom: 15px; color: white; min-height: 120px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .hero-section h1 { font-size: 2.2rem !important; margin: 0 !important; line-height: 1.2 !important; }
    .data-card {
        background: white; padding: 15px; border-radius: 12px;
        border-left: 6px solid #1e40af; border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 10px;
    }
    .data-card h4 { color: #1e40af; font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    div.stButton > button {
        background: #1e40af !important; color: white !important;
        border-radius: 10px !important; font-weight: 700 !important;
        height: 3.5em !important; width: 100% !important;
    }
    div.stDownloadButton > button {
        background-color: #10b981 !important; color: white !important;
        font-size: 1.1rem !important; font-weight: 800 !important;
        height: 3.8em !important; border-radius: 10px !important; width: 100% !important;
        box-shadow: 0 8px 15px rgba(16, 185, 129, 0.3) !important;
    }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ===============================
# PDF ENGINE
# ===============================
def generate_matching_pdf(details, sev):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFillColor(colors.HexColor("#1e40af"))
    p.rect(0, 725, 612, 60, fill=True, stroke=False)
    
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(306, 755, "BONE FRACTURE DIAGNOSTIC REPORT")
    p.setFont("Helvetica", 11)
    p.drawCentredString(306, 738, f"• Clinical Result")

    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, 690, "Patient Demographics")
    p.setFont("Helvetica", 11)
    p.drawString(60, 675, f"• Name: {details.get('Name', 'N/A')}")
    p.drawString(60, 655, f"• Age: {details['Age']}")
    p.drawString(60, 635, f"• Gender: {details['Gender']}")
    p.drawString(60, 615, f"• BMI Index: {details['BMI']}")
    p.drawString(60, 595, f"• Diabetes Status: {details['Diabetes']}")

    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, 555, "Injury Analysis")
    p.setFont("Helvetica", 11)
    p.drawString(60, 535, f"• Accident Type: {details['Accident']}")
    p.drawString(60, 515, f"• Impact Force: {details['Impact']}")
    p.drawString(60, 495, f"• Bone Involved: {details['Bone']}")
    p.drawString(60, 475, f"• Tissue Damage: {details['Soft']}")
    p.drawString(60, 455, f"• Swelling: {details['Swelling']}")
    p.drawString(60, 435, f"• Displacement: {details['Disp']} mm")
    p.drawString(60, 415, f"• Pain Score (VAS): {details['Pain']}")

    p.setFillColor(colors.HexColor("#1e40af"))
    p.setFont("Helvetica-Bold", 15)
    p.drawString(50, 350, f"PREDICTED SEVERITY: {sev.upper()}")
    
    p.setStrokeColor(colors.lightgrey)
    p.line(50, 60, 550, 60)
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColor(colors.grey)
    p.drawString(50, 45, "Fracture AI Engine | Validated Decision Support | 2026")
    
    p.save()
    buffer.seek(0)
    return buffer

# ===============================
# DATA & MODEL (FIXED FOR STRING ERRORS)
# ===============================
@st.cache_data
def load_and_train():
    df = pd.read_csv("injury1.csv")
    
    # 1. Separate Numerical and Categorical Columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 2. Impute Missing Numerical Values (Median)
    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    # 3. Impute Missing Categorical Values (Most Frequent)
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # 4. Encode Categorical Strings to Numbers
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Ensure data is string before encoding to avoid type comparison errors
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop("Severity", axis=1)
    y = df["Severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    return model, label_encoders, X.columns

model, encoders, features = load_and_train()

# ===============================
# UI LAYOUT
# ===============================
st.markdown("""
<div class="hero-section">
    <h1>BONE FRACTURE SEVERITY PREDICTION</h1>
    <p>Predictive Clinical Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("<div class='data-card'><h4>🧬 Patient Information</h4>", unsafe_allow_html=True)
    u_name = st.text_input("Patient Full Name", "")
    v1, v2 = st.columns(2)
    age = v1.number_input("Patient Age", 0, 120, 45)
    gender = v2.selectbox("Gender", encoders["Gender"].classes_)
    bmi = v1.number_input("BMI", 10.0, 50.0, 24.5)
    diabetes = v2.selectbox("Diabetes History", encoders["Diabetes"].classes_)
    pain = st.select_slider("Pain Score (VAS)", options=list(range(11)), value=5)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='data-card'><h4>📋 Medical Details</h4>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    accident = d1.selectbox("Accident Type", encoders["Accident_Type"].classes_)
    impact = d2.selectbox("Impact Force", encoders["Impact_Force"].classes_)
    bone = d1.selectbox("Bone Involved", encoders["Bone_Involved"].classes_)
    soft = d2.selectbox("Soft Tissue Injury", encoders["Soft_Tissue_Injury"].classes_)
    
    # --- NEW SWELLING INPUT ---
    swelling = d1.selectbox("Swelling Present", encoders["Swelling"].classes_)
    
    disp = d2.number_input("Displacement (mm)", 0.0, 50.0, 2.0)
    lines = d1.number_input("Fracture Lines", 1, 10, 1)
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("⚡ PREDICT"):
    # Encode all categorical user inputs
    input_data = {
        "Age": age, 
        "Gender": encoders["Gender"].transform([gender])[0],
        "BMI": bmi, 
        "Diabetes": encoders["Diabetes"].transform([diabetes])[0],
        "Accident_Type": encoders["Accident_Type"].transform([accident])[0],
        "Impact_Force": encoders["Impact_Force"].transform([impact])[0],
        "Time_To_Hospital_Hrs": 2.0, 
        "Swelling": encoders["Swelling"].transform([swelling])[0], # Swelling added here
        "Pain_Score": pain,
        "Bone_Involved": encoders["Bone_Involved"].transform([bone])[0],
        "Soft_Tissue_Injury": encoders["Soft_Tissue_Injury"].transform([soft])[0],
        "Fracture_Displacement_mm": disp, 
        "Fracture_Lines": lines
    }
    
    # Ensure columns are in same order as the model training data
    input_df = pd.DataFrame([input_data])[features]
    res_idx = model.predict(input_df)[0]
    
    st.session_state.severity = encoders["Severity"].inverse_transform([res_idx])[0]
    st.session_state.details = {
        "Name": u_name,
        "Age": age, "Gender": gender, "BMI": bmi, "Diabetes": diabetes,
        "Accident": accident, "Impact": impact, "Bone": bone,
        "Soft": soft, "Swelling": swelling, "Disp": disp, "Lines": lines, "Pain": pain
    }
    st.session_state.executed = True

if st.session_state.executed:
    st.markdown("---")
    sev = st.session_state.severity
    if sev == "Severe": st.error(f"### **STATUS: {sev.upper()} SEVERITY**")
    elif sev == "Mild": st.success(f"### **STATUS: {sev.upper()} SEVERITY**")
    else: st.warning(f"### **STATUS: {sev.upper()} SEVERITY**")

    pdf_file = generate_matching_pdf(st.session_state.details, sev)
    st.download_button(
        label="📥 DOWNLOAD PREDICTED RESULT",
        data=pdf_file,
        file_name=f"Report_{u_name.replace(' ', '_') if u_name else 'Patient'}.pdf",
        mime="application/pdf"
    )

st.markdown('<p style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:15px;">Fracture AI Engine | Clinical Decision Support 2026</p>', unsafe_allow_html=True)