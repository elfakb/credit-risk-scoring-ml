import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kredi Başvuru Değerlendirme", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "..", "models", "best_model.joblib"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "..", "models", "preprocessor.joblib"))
    with open(os.path.join(BASE_DIR, "..", "models", "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model, preprocessor, meta

model, preprocessor, meta = load_artifacts()
numeric_features = meta["numeric_features"]
categorical_features = meta["categorical_features"]
threshold = meta["threshold"]

st.title("🏦 Kredi Başvuru Değerlendirme Sistemi")
st.caption(f"Model: {meta['best_model_name']}  |  Test AUC: {meta['auc_scores'][meta['best_model_name']]:.3f}")
st.markdown("---")


st.subheader("👤 Kişisel Bilgiler")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Yaşınız", min_value=18, max_value=90, value=30)
with col2:
    home_ownership = st.selectbox(
        "Ev Sahiplik Durumunuz",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        format_func=lambda x: {
            "RENT": "Kirada oturuyorum",
            "OWN": "Kendi evim var",
            "MORTGAGE": "İpotekli evim var",
            "OTHER": "Diğer"
        }[x]
    )


st.subheader("Gelir ve İstihdam Bilgileri")
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Yıllık Geliriniz (TL)", min_value=0, value=60000, step=1000)
with col2:
    emp_length = st.number_input("Kaç yıldır çalışıyorsunuz?", min_value=0, max_value=60, value=5)



st.subheader("Kredi Talebi")
col1, col2 = st.columns(2)
with col1:
    loan_amnt = st.number_input("Talep Edilen Kredi Tutarı (TL)", min_value=500, value=10000, step=500)
    loan_intent = st.selectbox(
        "Kredi Amacınız",
        options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        format_func=lambda x: {
            "PERSONAL": "Kişisel Harcama",
            "EDUCATION": "Eğitim",
            "MEDICAL": "Sağlık",
            "VENTURE": "İş Kurma / Girişim",
            "HOMEIMPROVEMENT": "Ev Tadilatı",
            "DEBTCONSOLIDATION": "Borç Birleştirme"
        }[x]
    )
with col2:
    st.caption("")
    loan_grade = st.select_slider(
        "Kredi Notu (Risk Derecesi)",
        options=["A", "B", "C", "D", "E", "F", "G"],
        value="C",
        help="A: En düşük risk, G: En yüksek risk"
    )
   
    grade_rate_map = {"A": 7.5, "B": 10.0, "C": 13.0, "D": 15.5, "E": 17.5, "F": 19.5, "G": 21.5}
    int_rate = st.slider("Faiz Oranı (%)", 5.0, 25.0, grade_rate_map[loan_grade], 0.1)

loan_percent_income = round(loan_amnt / income, 4) if income > 0 else 0
st.info(f"Kredi tutarının gelirinize oranı: **%{loan_percent_income*100:.1f}**")



st.subheader("Kredi Geçmişi")
col1, col2 = st.columns(2)
with col1:
    default_on_file = st.radio(
        "Daha önce bir kredi ödemesinde ciddi gecikmeniz oldu mu?",
        options=["N", "Y"],
        format_func=lambda x: "Hayır" if x == "N" else "Evet",
        horizontal=True
    )
with col2:
    cred_hist_length = st.number_input("Kaç yıldır kredi geçmişiniz var?", min_value=0, max_value=60, value=5)

st.markdown("---")



credit_history_ratio = cred_hist_length / age
emp_length_ratio = emp_length / age
loan_to_emp_length = loan_amnt / (emp_length + 1)
income_to_loan = income / (loan_amnt + 1)
has_prior_default = 1 if default_on_file == "Y" else 0

if age <= 25:
    age_group = "18-25"
elif age <= 35:
    age_group = "26-35"
elif age <= 50:
    age_group = "36-50"
else:
    age_group = "50+"

row = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cred_hist_length,
    "credit_history_ratio": credit_history_ratio,
    "emp_length_ratio": emp_length_ratio,
    "loan_emp_length": loan_to_emp_length,
    "income_to_loan": income_to_loan,
    "default": has_prior_default,
    "person_home_ownership": home_ownership,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "cb_person_default_on_file": default_on_file,
    "age_group": age_group,
}])[numeric_features + categorical_features]



row_processed = preprocessor.transform(row)
proba_default = model.predict_proba(row_processed)[0, 1]
decision = proba_default >= threshold



if st.button("Kredi Değerlendirmesini Yap", type="primary"):
    st.subheader("Sonuç")

    if decision:
        st.error("❌ KREDİ ONAYLANMADI")
    else:
        st.success("✅ KREDİ ONAYLANDI")

    st.metric("Tahmini Temerrüt (Default) Riski", f"%{proba_default*100:.1f}")
    st.progress(min(float(proba_default), 1.0))
    st.caption(f"Karar eşiği: %{threshold*100:.0f}. Bu değerin üzerindeki riskler onaylanmıyor.")

st.markdown("---")
st.caption("")