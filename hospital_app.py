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
