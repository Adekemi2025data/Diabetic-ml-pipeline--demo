import streamlit as st

st.title("Natural Language Interface for Diabetes Dataset")

question = st.text_input("Ask a question about the dataset:")

if question:
    st.write(f"You asked: {question}")
    st.write("This means your Streamlit app is working!")
