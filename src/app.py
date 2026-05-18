# src/app.py
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import streamlit as st
from src.nl_pipeline import handle_query

st.set_page_config(page_title="Diabetes Risk Assistant", page_icon="🩺")

st.title("🩺 Diabetes Risk Assistant")
st.write("Ask a question about diabetes risk, symptoms, or your dataset.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = handle_query(query)
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
