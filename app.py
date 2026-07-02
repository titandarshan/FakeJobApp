import streamlit as st
import joblib
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Job Fraud Detector", page_icon="🛡️", layout="wide")

# --- 2. LOAD AI MODELS ---
@st.cache_resource
def load_components():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('numeric_scaler.pkl')
    model = joblib.load('fake_job_model.pkl')
    return tfidf, scaler, model

try:
    tfidf, scaler, model = load_components()
except Exception as e:
    st.error("⚠️ Error loading models. Please run train_model.py first!")
    st.stop()

# --- 3. UI HEADER ---
st.title("🛡️ Fake Job Posting Detection System")
st.markdown("Evaluate the legitimacy of a job posting by analyzing text patterns and structured metadata.")
st.divider()

# --- 4. USER INPUT LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Job Description Text")
    job_text = st.text_area(
        "Paste the full job posting here (Title, Profile, Requirements):", 
        height=280,
        placeholder="e.g., URGENT hiring! Work from home and make $5000 a week. Send bank details..."
    )

with col2:
    st.subheader("📊 Structured Metadata")
    st.markdown("Check the boxes that apply to this job:")
    has_salary = st.checkbox("💰 Salary is specified")
    profile_missing = st.checkbox("🏢 Company Profile is missing")
    telecommuting = st.checkbox("🏠 Remote/Telecommuting job")
    has_company_logo = st.checkbox("🖼️ Has a Company Logo")
    has_questions = st.checkbox("❓ Includes Screening Questions")

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("🚀 Analyze Job Posting", type="primary", use_container_width=True):
    if not job_text.strip():
        st.warning("⚠️ Please enter the job description text to begin analysis.")
    else:
        with st.spinner("Processing Natural Language and Metadata..."):
            
            # Process inputs
            text_vectorized = tfidf.transform([job_text]).toarray()
            text_len = len(job_text)
            num_features = np.array([[
                text_len, int(has_salary), int(profile_missing), 
                int(telecommuting), int(has_company_logo), int(has_questions)
            ]])
            
            num_scaled = scaler.transform(num_features)
            final_features = np.hstack((text_vectorized, num_scaled))
            
            # Make Prediction
            prediction = model.predict(final_features)
            probability = model.predict_proba(final_features)[0]
            
            # --- 6. RESULTS DISPLAY ---
            st.subheader("🎯 Analysis Results")
            res_col1, res_col2 = st.columns(2)
            
            if prediction[0] == 1:
                res_col1.error("🚨 **FLAGGED: LIKELY FRAUDULENT**")
                res_col1.write(f"Confidence Score: **{probability[1]:.2%}**")
                res_col2.info("💡 **Why?** The model detected linguistic patterns or metadata highly correlated with past scams.")
            else:
                res_col1.success("✅ **CLEARED: LIKELY LEGITIMATE**")
                res_col1.write(f"Confidence Score: **{probability[0]:.2%}**")
                res_col2.info("💡 **Why?** The text structure and metadata align with standard, verified industry job postings.")