import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
  page_title="Smart Hospital Patient Navigator",
  page_icon="🏥",
  layout="wide"
)

# Hero HEader
st.markdown("""
<div>
  <h1>AI Smart Patient Navigator</h1>
</div>
""", unsafe_allow_html=True)


# Load Model
def load_model():
  with open('hospital_model_rayyan.pkl', 'rb') as f:
    return pickle.load(f)

bundle = load_model()
model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']
column_to_scale = bundle['column_to_scale']
department_map_inverted = bundle['department_map_inverted']
gender_map = bundle['gender_map']
temperature_map = bundle['temperature_map']
heart_rate_map = bundle['heart_rate_map']
duration_map = bundle['duration_map']
chief_complaint_map = bundle['chief_complaint_map']

DEPT_INFO = {
  'Respiratory Medicine' : {
    'icon':'🫁',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Specialises in condition affecting the lungs and airways',
    'next':['Visit Level 2', 'Wing B', 'Estimated wait 15-25 min', 'Please wear a mask']
  },
    'Cardiology' : {
    'icon':'♥️',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Specialises in condition affecting heart and cardiovascular conditions.',
    'next':['Visit Level 3', 'Wing A', 'Estimated wait 20-30 min', 'Bring any previous ECG reports']
  },
    'Gastroenterology' : {
    'icon':'🫃',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Specialises in digestive sytem and abdominal conditions',
    'next':['Visit Level 1', 'Wing C', 'Estimated wait 10-20 min', 'Avoid eating before consultation']
  },
    'Neurology' : {
    'icon':'🧠',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Specialises in brain, spine, and nervous system conditions',
    'next':['Visit Level 4', 'Wing A', 'Estimated wait 25-35 min', 'Bring list of current medications']
  },
    'General Medicine' : {
    'icon':'🩺',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Handles general health concerns and non-specialist conditions.',
    'next':['Visit Level 1', 'Wing A', 'Estimated wait 10-15 min', 'Registration desk is open 24/7']
  },
    'Dermatology' : {
    'icon':'🦠',
    'color':'#0284C7',
    'bg':'#E0F2FE',
    'border':'#7DD3FC',
    'desc':'Specialises in skin, hair and nail conditions',
    'next':['Visit Level 2', 'Wing D', 'Estimated wait 15-20 min', 'Bring photos of affected area if possible']
  }
}

with st.form("triage_form"):
  # Symptomps
  c1, c2, c3, c4 = st.columns(4)
  with c1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
  with c2:
    headache = st.checkbox("Headache")
    chest_pain = st.checkbox("Chest Pain")
  with c3:
    stomach_pain = st.checkbox("Stomach Pain")
    shortness_breath = st.checkbox("Shortness Breath")
  with c4:
    nausea_vomitting = st.checkbox("Nausea/Vomiting")
    dizziness = st.checkbox("Dizziness")

  c5, _, _, _ = st.columns(4)
  with c5:
    skin_rash = st.checkbox("Skin Rash")

  # Duration and Complaints
  col_cc, col_dur = st.columns(2)
  with col_cc:
    chief_complaint = st.selectbox("Chief Complaint", options=list(chief_complaint_map.keys()))
  with col_dur:
    duration = st.selectbox("Duration", options=list(duration_map.keys()), index=1)

  # Severity
  col_temp, col_hr = st.columns(2)
  with col_temp:
    temperature_level = st.selectbox("Temperature", options=list(temperature_map.keys()), index=1)
  with col_hr:
    heart_rate_level = st.selectbox("Heart Rate Level", options=list(heart_rate_map.keys()), index=1)

  # Medical History
  ch1, ch2, ch3 = st.columns(3)
  with ch1:
    hypertension = st.checkbox("Hypertension")
  with ch2:
    heart_disease = st.checkbox("Heart Disease")
  with ch3:
    asthma = st.checkbox("Asthma")

  # Patient Info
  col_age, col_gen = st.columns(2)
  with col_age:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
  with col_gen:
    gender = st.selectbox("Gender", options=['Female', 'Male'])
  
  submitted = st.form_submit_button("Get AI Recomendation")

# Get Result
if submitted:
  patient_info = {
    'age':age,
    'gender':gender_map.get(gender, 0),
    'fever':int(fever),
    'cough':int(cough),
    'headache':int(headache),
    'chest_pain':int(chest_pain),
    'stomach_pain':int(stomach_pain),
    'shortness_breath':int(shortness_breath),
    'nausea_vomiting':int(nausea_vomiting),
    'dizziness':int(dizziness),
    'skin_rash':int(skin_rash),
    'temperature_level':temperature_map.get(temperature_level, 1),
    'heart_rate_level':heart_rate_map.get(heart_rate_level, 1),
    'duration':duration_map.get(duration, 1),
    'asthma':int(asthma),
    'hypertension':int(hypertension),
    'heart_disease':int(heart_disease),
    'chief_complaint': chief_complaint_map.get(chief_complaint, 9)
  }
  
  patient = pd.DataFrame([patient_info])

  patient_scaled = patient.copy()
  patient_scaled[column_to_scale] = scaler.transform(patient.[column_to_scale])

  pred = model.predict(patient_scaled[features])[0]
  proba = model.predict_proba(patient_scaled[features])[0]
  dept_name = department_map_inverted[pred]
  confidence = proba[pred] * 100
  info = DEPT_INFO[dept_name]

  st.markdown(
    """
    <div>
    <h2>{dept_name}</h2>
    </div>
    """, unsafe_allow_html=True
  )
