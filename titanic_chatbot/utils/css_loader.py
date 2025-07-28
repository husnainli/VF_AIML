import streamlit as st

def load_css(file_path: str):
    """Inject CSS into Streamlit app from a file."""
    with open(file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
